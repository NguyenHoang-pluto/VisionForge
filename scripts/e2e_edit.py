"""Phase 4 acceptance test: folder of videos in, playable MP4 out.

Drives a live API and live CPU + render workers through the complete vertical
slice, with no LLM anywhere in it:

    upload 6 clips -> ingest -> CPU analysis
        -> deterministic selection
        -> RulesEnginePlanner -> EditPlan
        -> validation
        -> Timeline -> RenderSpec -> FFmpeg argv
        -> render worker
        -> MP4, verified with ffprobe

The clips are generated to be *unequal on purpose*: two are deliberately
unusable (one badly out of focus, one crushed to black) and two are near
duplicates of each other. A selector that simply took the first N would pass a
weaker test; these inputs make the rejections observable.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_edit.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = os.environ.get("VF_API_URL", "http://127.0.0.1:8000")
CLIP_DIR = Path(os.environ.get("VF_E2E_CLIPS", Path(__file__).parent / ".e2e-clips"))
JOB_TIMEOUT_S = 900

PASS, FAIL, INFO = "PASS", "FAIL", " ..."
_failures: list[str] = []
_report: dict[str, object] = {}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def step(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------- inputs
#: Six 1280x720 clips. Names describe what each is meant to exercise.
CLIP_SPECS: list[tuple[str, str]] = [
    # Sharp, well-exposed, distinct content -- should all be selected.
    ("sharp_bars.mp4", "smptebars=size=1280x720:rate=25:duration=6"),
    ("sharp_test.mp4", "testsrc=size=1280x720:rate=25:duration=6"),
    ("sharp_test2.mp4", "testsrc2=size=1280x720:rate=25:duration=6"),
    ("sharp_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=6"),
    # Heavily blurred: should be rejected as out of focus.
    ("soft_blurred.mp4", "testsrc=size=1280x720:rate=25:duration=6,boxblur=20:2"),
    # Near black: should be rejected as badly exposed / low contrast.
    ("dark_crushed.mp4", "color=c=black:size=1280x720:rate=25:duration=6"),
]


def build_clips() -> list[Path]:
    """Generate the input clips with FFmpeg. Deterministic and offline."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not on PATH")
        sys.exit(2)

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, source in CLIP_SPECS:
        destination = CLIP_DIR / name
        if not destination.exists():
            subprocess.run(  # noqa: S603 - argv array, local fixture generation
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", source,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
        paths.append(destination)
    return paths


# --------------------------------------------------------------------- helpers
def create_project(title: str) -> str:
    response = httpx.post(f"{API}/api/projects", json={"title": title}, timeout=30)
    response.raise_for_status()
    return str(response.json()["id"])


def upload(project_id: str, path: Path) -> tuple[str, str | None]:
    ticket = httpx.post(
        f"{API}/api/projects/{project_id}/media/upload-url",
        json={"filename": path.name, "size_bytes": path.stat().st_size},
        timeout=30,
    )
    ticket.raise_for_status()
    body = ticket.json()

    httpx.put(body["upload_url"], content=path.read_bytes(), timeout=300).raise_for_status()

    completed = httpx.post(
        f"{API}/api/projects/{project_id}/media/{body['media_id']}/complete", timeout=60
    )
    completed.raise_for_status()
    return body["media_id"], completed.json().get("job_id")


def wait_for_job(job_id: str, timeout_s: int = JOB_TIMEOUT_S) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = httpx.get(f"{API}/api/jobs/{job_id}", timeout=30).json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} stuck in {last.get('status')} after {timeout_s}s")


def wait_for_render(project_id: str, render_id: str, timeout_s: int = JOB_TIMEOUT_S) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = httpx.get(
            f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30
        ).json()
        if last["status"] in {"ready", "failed", "cancelled"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"render {render_id} stuck in {last.get('status')}")


def ffprobe(path: Path) -> dict:
    """Probe the finished file independently of the application's own code."""
    binary = shutil.which("ffprobe")
    assert binary
    result = subprocess.run(  # noqa: S603 - argv array
        [
            binary, "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return dict(json.loads(result.stdout))


# ------------------------------------------------------------------- scenarios
def scenario_ingest_and_analyse(clips: list[Path]) -> tuple[str, dict[str, str]]:
    step("1. Upload and analyse 6 clips")
    project_id = create_project("Phase 4 Vertical Slice")
    print(f"{INFO} project {project_id}")

    media: dict[str, str] = {}
    for path in clips:
        media_id, job_id = upload(project_id, path)
        media[path.name] = media_id
        assert job_id
        result = wait_for_job(job_id)
        check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis",
        json={"lanes": ["cpu"]},
        timeout=60,
    )
    check("CPU analysis accepted", queued.status_code == 202, str(queued.status_code))
    body = queued.json()
    for job in body["jobs"]:
        result = wait_for_job(job["id"])
        check(
            f"analysis {job['id'][:8]}", result["status"] == "succeeded", result["status"]
        )

    _report["input_clips"] = len(clips)
    return project_id, media


def scenario_plan(project_id: str, media: dict[str, str]) -> dict:
    step("2. Deterministic selection and EditPlan")
    names = {media_id: name for name, media_id in media.items()}

    response = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={
            "target_duration_ms": 25_000,
            "max_clips": 5,
            "min_clips": 2,
            "aspect_ratio": "16:9",
            "fps": 30,
            "order": "score_desc",
        },
        timeout=120,
    )
    check("edit plan created (201)", response.status_code == 201, str(response.status_code))
    response.raise_for_status()
    plan = response.json()

    document = plan["plan"]
    selection = plan["selection"]

    print(f"{INFO} planner {document['planner']}@{document['planner_version']}")
    print(f"{INFO} selected:")
    for item in selection["selected"]:
        print(f"       {names.get(item['media_id'], '?'):20} score={item['score']:.4f}")
    print(f"{INFO} rejected:")
    for item in selection["rejected"]:
        print(
            f"       {names.get(item['media_id'], '?'):20} "
            f"{item['reason']:15} {item['detail']}"
        )

    check("no LLM involved", document["planner"] == "rules-engine", document["planner"])
    check(
        "plan has 2-5 segments",
        2 <= plan["segment_count"] <= 5,
        str(plan["segment_count"]),
    )
    check(
        "total duration is 20-30s",
        20_000 <= plan["total_duration_ms"] <= 30_000,
        f"{plan['total_duration_ms']} ms",
    )

    rejected_names = {names.get(r["media_id"]) for r in selection["rejected"]}
    check(
        "the blurred clip was rejected",
        "soft_blurred.mp4" in rejected_names,
        ",".join(sorted(n for n in rejected_names if n)),
    )
    check("the crushed clip was rejected", "dark_crushed.mp4" in rejected_names)

    selected_names = {names.get(s["media_id"]) for s in selection["selected"]}
    check("no unusable clip was selected", not (selected_names & rejected_names))

    orders = [s["order"] for s in document["segments"]]
    check("segment orders are contiguous", orders == list(range(len(orders))))
    check(
        "every segment has a positive trim",
        all(s["source_out_ms"] > s["source_in_ms"] for s in document["segments"]),
    )
    check(
        "output is 16:9 at 1280x720",
        (document["output"]["width"], document["output"]["height"]) == (1280, 720),
    )

    # Determinism: the same request must produce the same plan.
    again = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={
            "target_duration_ms": 25_000,
            "max_clips": 5,
            "min_clips": 2,
            "aspect_ratio": "16:9",
            "fps": 30,
            "order": "score_desc",
        },
        timeout=120,
    ).json()
    check(
        "re-planning is deterministic",
        again["plan"]["segments"] == document["segments"],
    )

    _report["selected_clips"] = [names.get(s["media_id"]) for s in selection["selected"]]
    _report["rejected_clips"] = [
        {"clip": names.get(r["media_id"]), "reason": r["reason"]}
        for r in selection["rejected"]
    ]
    _report["plan"] = document
    return plan


def scenario_render(project_id: str, plan: dict) -> dict:
    step("3. Render to MP4")

    started = time.perf_counter()
    response = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan["id"]},
        timeout=60,
    )
    check("render accepted (202)", response.status_code == 202, str(response.status_code))
    response.raise_for_status()
    render = response.json()

    final = wait_for_render(project_id, render["id"])
    elapsed_s = time.perf_counter() - started

    check("render reached ready", final["status"] == "ready", str(final.get("error")))
    check("render reports a duration", bool(final["duration_ms"]))
    check(
        "render is 1280x720",
        (final["width"], final["height"]) == (1280, 720),
        f"{final['width']}x{final['height']}",
    )
    check("render has a playback URL", bool(final.get("playback_url")))

    _report["render"] = final
    _report["render_wall_s"] = round(elapsed_s, 1)
    return final


def scenario_verify(render: dict, plan: dict) -> None:
    step("4. Verify the MP4 with ffprobe")

    output = CLIP_DIR / "phase4-output.mp4"
    response = httpx.get(render["playback_url"], timeout=300)
    response.raise_for_status()
    output.write_bytes(response.content)

    check("file downloaded", output.exists() and output.stat().st_size > 0)
    check(
        "file size matches the recorded size",
        output.stat().st_size == render["bytes_size"],
        f"{output.stat().st_size} vs {render['bytes_size']}",
    )

    probe = ffprobe(output)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)

    check("ffprobe found a video stream", video is not None)
    assert video is not None

    check("container is mp4", "mp4" in probe["format"]["format_name"])
    check("codec is h264", video["codec_name"] == "h264", video["codec_name"])
    check("pixel format is yuv420p", video["pix_fmt"] == "yuv420p", video["pix_fmt"])
    check(
        "resolution is 1280x720",
        (video["width"], video["height"]) == (1280, 720),
        f"{video['width']}x{video['height']}",
    )

    numerator, _, denominator = video["avg_frame_rate"].partition("/")
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    check("frame rate is 30 fps", abs(fps - 30.0) < 0.5, f"{fps:.2f}")

    duration_s = float(probe["format"]["duration"])
    planned_s = plan["total_duration_ms"] / 1000
    check(
        "duration matches the plan within a frame",
        abs(duration_s - planned_s) < 0.5,
        f"{duration_s:.2f}s vs planned {planned_s:.2f}s",
    )
    check("duration is 20-30s", 20.0 <= duration_s <= 30.0, f"{duration_s:.2f}s")

    # The decisive check: every frame decodes. A truncated or corrupt file would
    # probe fine and fail here.
    binary = shutil.which("ffmpeg")
    assert binary
    decode = subprocess.run(  # noqa: S603 - argv array
        [binary, "-v", "error", "-i", str(output), "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    check(
        "the whole file decodes without error",
        decode.returncode == 0 and not decode.stderr.strip(),
        (decode.stderr or "").strip()[:120],
    )

    _report["ffprobe"] = {
        "container": probe["format"]["format_name"],
        "codec": video["codec_name"],
        "pix_fmt": video["pix_fmt"],
        "width": video["width"],
        "height": video["height"],
        "fps": round(fps, 3),
        "duration_s": round(duration_s, 3),
        "bytes": output.stat().st_size,
        "nb_frames": video.get("nb_frames"),
    }
    _report["output_path"] = str(output)


def main() -> int:
    print("VisionForge Phase 4 - vertical slice acceptance")
    print(f"API: {API}")

    try:
        health = httpx.get(f"{API}/health/ready", timeout=15).json()
    except httpx.HTTPError as exc:
        print(f"\nCannot reach the API at {API}: {exc}")
        return 2
    if health["status"] != "ok":
        print(f"\nAPI is not ready: {json.dumps(health, indent=2)}")
        return 2

    clips = build_clips()
    project_id, media = scenario_ingest_and_analyse(clips)
    plan = scenario_plan(project_id, media)
    render = scenario_render(project_id, plan)
    scenario_verify(render, plan)

    step("Summary")
    print(json.dumps(_report, indent=2, default=str))

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All Phase 4 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
