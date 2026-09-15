"""The render pipeline: plan -> timeline -> spec -> FFmpeg -> object storage.

Steps: PREPARE -> COMPILE -> RENDER -> PUBLISH -> FINALIZE.

Runs on the ``render`` queue, provisioned in Phase 1 and unused until now. It is
separate from ``cpu`` for the reason Phase 0 gave: an encode takes minutes and
must not sit behind a queue of second-long analysis tasks, nor block them.

Everything here is CPU and FFmpeg. No GPU, no models — Phase 4 uses the card for
nothing, leaving it entirely to the Phase 3 intelligence lanes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from visionforge.domain.editplan import EditPlan, MediaFact, assert_valid, plan_from_payload
from visionforge.domain.errors import PermanentError, TransientError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import DerivativeKind, MediaKind, MediaStatus
from visionforge.domain.render import RenderStatus, render_key
from visionforge.domain.timeline import build_render_spec, compile_timeline
from visionforge.infra.db.models import EditPlanRow, MediaAsset, MediaDerivative, RenderRow
from visionforge.infra.ffmpeg import probe_media
from visionforge.infra.ffmpeg.compiler import compile_render_argv
from visionforge.infra.ffmpeg.runner import FFMPEG, resolve_binary
from visionforge.infra.storage import S3ObjectStore
from visionforge.workers.runtime import JobContext

logger = logging.getLogger(__name__)

#: Wall-clock ceiling for one encode. A 30-second 720p output on a 6-core laptop
#: takes seconds; 30 minutes means something is pathologically wrong and the job
#: should fail rather than occupy the queue indefinitely.
RENDER_TIMEOUT_S = 1800.0


def _render_row(ctx: JobContext) -> RenderRow:
    render_id = ctx.job.params.get("render_id")
    if not render_id:
        raise PermanentError("render job has no render_id")
    row = ctx.session.get(RenderRow, UUID(str(render_id)))
    if row is None:
        raise PermanentError("render row disappeared")
    return row


def _store(ctx: JobContext) -> S3ObjectStore:
    store = ctx.data.get("store")
    if store is None:
        store = S3ObjectStore()
        ctx.data["store"] = store
    return store


def _workdir(ctx: JobContext) -> str:
    workdir = ctx.data.get("workdir")
    if not workdir or not os.path.isdir(workdir):
        workdir = tempfile.mkdtemp(prefix=f"vfr-{ctx.job_id}-")
        ctx.data["workdir"] = workdir
    return str(workdir)


# ---------------------------------------------------------------------- PREPARE
def step_prepare(ctx: JobContext) -> None:
    """Re-validate the plan and pull every source it references.

    The plan is validated **again** here, against the database as it is now.
    It was validated when it was created, but media can be deleted between
    planning and rendering, and a stale plan must not reach FFmpeg. Validation
    is cheap; a half-finished render against missing media is not.

    Sources come from the 720p proxy where one exists — the same proxy-first
    rule as analysis. Rendering a preview from 4K masters would decode eight
    times the pixels to produce a 720p output.
    """
    render = _render_row(ctx)
    plan_row = ctx.session.get(EditPlanRow, render.edit_plan_id)
    if plan_row is None:
        raise PermanentError("edit plan disappeared")

    plan = plan_from_payload(plan_row.plan)
    facts, sources = _resolve_media(ctx, plan)

    # The gate. Nothing past this point can be reached by an invalid plan.
    assert_valid(plan, facts)

    store = _store(ctx)
    workdir = _workdir(ctx)
    local_paths: dict[MediaId, str] = {}
    for media_id, (storage_key, suffix) in sources.items():
        destination = os.path.join(workdir, f"src-{media_id}{suffix}")
        if not os.path.exists(destination):
            store.download_to(storage_key, destination)
        local_paths[media_id] = destination

    ctx.data["plan"] = plan
    ctx.data["local_paths"] = local_paths
    render.status = RenderStatus.RENDERING


#: Fallback extension when a storage key carries none, by kind. FFmpeg sniffs
#: content rather than trusting a name, but a *wrong* extension is worse than a
#: missing one -- handing it ``track.mp4`` for an MP3 makes the demuxer's first
#: guess the wrong one.
_DEFAULT_SUFFIX: dict[MediaKind, str] = {
    MediaKind.VIDEO: ".mp4",
    MediaKind.AUDIO: ".mp3",
    MediaKind.IMAGE: ".jpg",
}


def _resolve_media(
    ctx: JobContext, plan: EditPlan
) -> tuple[dict[MediaId, MediaFact], dict[MediaId, tuple[str, str]]]:
    """Look up each referenced asset and choose which representation to read.

    Returns the facts the validator needs and, separately, the storage key to
    download. Keeping them apart matters: the validator must never see a storage
    key, or a future check could start depending on one -- and the plan, which a
    client can influence, never carries either.

    Covers the music bed as well as the clips (``referenced_media_ids``). The
    ownership check is still the validator's: the rows are fetched by id, the
    project each one actually belongs to becomes a ``MediaFact``, and
    ``assert_valid`` rejects a cue pointing at another project *before* this
    worker downloads a byte of it.
    """
    media_ids = list(plan.referenced_media_ids)
    rows = (
        ctx.session.execute(select(MediaAsset).where(MediaAsset.id.in_(media_ids))).scalars().all()
    )
    by_id = {MediaId(row.id): row for row in rows}

    facts: dict[MediaId, MediaFact] = {}
    sources: dict[MediaId, tuple[str, str]] = {}

    for media_id in media_ids:
        row = by_id.get(media_id)
        if row is None:
            continue  # validate_plan reports this as unknown_media

        kind = MediaKind(row.kind)

        # Proxy-first, for video only. Ingest never makes one for audio, so the
        # lookup would always miss -- but asking explicitly says why, and stops
        # a future audio derivative from silently becoming the render source.
        key = row.storage_key
        if kind is MediaKind.VIDEO:
            proxy = ctx.session.execute(
                select(MediaDerivative).where(
                    MediaDerivative.media_id == row.id,
                    MediaDerivative.kind == DerivativeKind.PROXY,
                )
            ).scalar_one_or_none()
            if proxy is not None:
                key = proxy.storage_key

        sources[media_id] = (key, Path(key).suffix or _DEFAULT_SUFFIX.get(kind, ".mp4"))
        facts[media_id] = MediaFact.from_media(
            media_id=media_id,
            project_id=ProjectId(row.project_id),
            kind=kind,
            status=MediaStatus(row.status),
            duration_ms=row.duration_ms,
            width=row.width,
            height=row.height,
        )
    return facts, sources


# ---------------------------------------------------------------------- COMPILE
def step_compile(ctx: JobContext) -> None:
    """Plan -> Timeline -> RenderSpec. Pure; no I/O, no FFmpeg."""
    render = _render_row(ctx)
    plan: EditPlan = ctx.data["plan"]

    timeline = compile_timeline(plan)
    output_path = os.path.join(_workdir(ctx), "output.mp4")
    spec = build_render_spec(timeline, local_paths=ctx.data["local_paths"], output_path=output_path)

    ctx.data["timeline"] = timeline
    ctx.data["spec"] = spec
    render.timeline = timeline.as_payload()
    render.spec = spec.as_payload()


# ----------------------------------------------------------------------- RENDER
def step_render(ctx: JobContext) -> None:
    """Run FFmpeg, streaming ``-progress`` so the bar reflects real encoding.

    The subprocess is an argv array with a hard timeout and no shell, matching
    the Phase 2 posture. Progress is parsed from ``out_time_us`` rather than
    estimated from elapsed time, because encode speed varies by an order of
    magnitude between a static shot and a handheld pan.
    """
    spec = ctx.data["spec"]
    args = compile_render_argv(spec)
    executable = resolve_binary(FFMPEG)

    started = time.perf_counter()
    process = subprocess.Popen(
        [executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        _consume_progress(ctx, process, total_ms=spec.duration_ms)
        _, stderr = process.communicate(timeout=RENDER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise TransientError(f"render exceeded {RENDER_TIMEOUT_S}s") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000

    if process.returncode != 0:
        tail = "\n".join((stderr or "").strip().splitlines()[-8:])
        raise PermanentError(
            f"ffmpeg exited with {process.returncode}",
            hint=tail or "No stderr output.",
        )

    output_path = spec.output_path
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise PermanentError("ffmpeg reported success but produced no output file")

    ctx.data["render_ms"] = elapsed_ms
    ctx.data["output_path"] = output_path


def _consume_progress(ctx: JobContext, process: subprocess.Popen[str], *, total_ms: int) -> None:
    """Read ``key=value`` progress lines and publish coarse updates.

    Published on change of whole percent, not per line: FFmpeg emits progress
    several times a second, and one Redis publish per line would cost more than
    the encode.
    """
    from visionforge.infra.ffmpeg.compiler import parse_progress_line, progress_fraction

    if process.stdout is None:
        return

    last_percent = -1
    for line in process.stdout:
        parsed = parse_progress_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key != "out_time_us":
            continue
        try:
            fraction = progress_fraction(int(value), total_ms)
        except ValueError:
            continue

        percent = int(fraction * 100)
        if percent != last_percent:
            last_percent = percent
            ctx.data["render_progress"] = fraction


# ---------------------------------------------------------------------- PUBLISH
def step_publish(ctx: JobContext) -> None:
    """Probe the finished file, then upload it.

    Probed before upload so that a corrupt output is caught here rather than
    discovered by a user pressing play. The measured values are what get stored:
    a render row that reports the duration it *intended* would hide exactly the
    discrepancies worth catching.
    """
    render = _render_row(ctx)
    output_path = ctx.data["output_path"]

    metadata = probe_media(output_path)
    if metadata.kind is not MediaKind.VIDEO:
        raise PermanentError("render output is not a video file")

    key = render_key(str(render.project_id), str(render.id))
    _store(ctx).upload_file(output_path, key, content_type="video/mp4")

    render.storage_key = key
    render.bytes_size = os.path.getsize(output_path)
    render.duration_ms = metadata.duration_ms
    render.width = metadata.width
    render.height = metadata.height
    render.fps = metadata.fps


# --------------------------------------------------------------------- FINALIZE
def step_finalize(ctx: JobContext) -> None:
    render = _render_row(ctx)
    render.status = RenderStatus.READY
    render.error = None
    render.metrics = {
        "render_ms": round(float(ctx.data.get("render_ms", 0.0)), 1),
        "input_count": len(ctx.data["spec"].inputs),
        "segment_count": len(ctx.data["spec"].segments),
        "output_bytes": render.bytes_size,
        "encoder": "libx264",
        "preset": ctx.data["spec"].preset,
        "crf": ctx.data["spec"].crf,
    }
    ctx.job.result = {
        "render_id": str(render.id),
        "storage_key": render.storage_key,
        "duration_ms": render.duration_ms,
        "width": render.width,
        "height": render.height,
        "fps": render.fps,
        "bytes": render.bytes_size,
    }
    cleanup(ctx)


def cleanup(ctx: JobContext) -> None:
    """Remove the scratch directory. Safe to call more than once."""
    ctx.data.pop("plan", None)
    ctx.data.pop("spec", None)
    ctx.data.pop("timeline", None)
    workdir = ctx.data.pop("workdir", None)
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


STEPS = {
    "PREPARE": step_prepare,
    "COMPILE": step_compile,
    "RENDER": step_render,
    "PUBLISH": step_publish,
    "FINALIZE": step_finalize,
}

__all__ = ["RENDER_TIMEOUT_S", "STEPS", "cleanup"]
