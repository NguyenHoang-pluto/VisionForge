"""Phase 10 acceptance test: an edit, changed in words, versioned, and rendered.

Phase 9 proved the editing primitives reach a playable MP4. This proves the
co-editor: that an edit which already exists can be *changed* by asking, that
the change is a validated delta rather than a regeneration, that every change is
a version, and that undo restores exactly what was there.

    upload + ingest + analyse   (the unchanged Phase 2/3 pipelines)
      -> plan                   version 1   (the unchanged rules engine)
      -> render                             (the unchanged render worker)
      -> hand-cut + subtitles   version 2   (the unchanged Phase 6/9 route)
      -> "faster opening, quieter music"
                                version 3   (Phase 10)
      -> render
      -> "undo"                 version 2 restored, byte for byte
      -> "remove the third clip and use bold subtitles"
                                version 4
      -> render                             -> MP4, verified with ffprobe

Four claims are under test.

**A change is a patch, not a regeneration.** The clips the user kept are the
same rows, trimmed the same way, except where the change said otherwise.

**A refused change changes nothing.** Invalid operations, an out-of-range
parameter and an unknown operation kind are each rejected with the edit left
byte-identical -- checked by reading the plan back, not by trusting the status
code.

**Undo restores, it does not recompute.** The plan after an undo is compared
byte for byte with the plan before the change.

**The render pipeline is untouched.** The final MP4 is verified independently:
streams, resolution, frame rate, duration against the plan's own arithmetic, and
a full decode.

Everything is generated with FFmpeg. No network, no model weights, no
copyrighted bytes.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_coedit.py

The co-editor's deterministic path needs no AI provider and is what carries the
assertions above. If one *is* configured (``LLM_ENABLED=true``, including the
local stub) the script additionally exercises the model path and says which
provider answered; with none, it checks that a request needing interpretation is
refused cleanly instead.

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
from typing import Any

import httpx

API = os.environ.get("VF_API_URL", "http://127.0.0.1:8000")
CLIP_DIR = Path(os.environ.get("VF_E2E_CLIPS", Path(__file__).parent / ".e2e-clips"))
JOB_TIMEOUT_S = 900

#: The edit this acceptance builds.
CLIP_MS = 4_000
SOURCE_S = 10

#: One frame at 30 fps is 33 ms; the muxer rounds, and a speed change rounds
#: again. Two frames of slack.
DURATION_TOLERANCE_MS = 250

PASS, FAIL = "PASS", "FAIL"
_failures: list[str] = []
_report: dict[str, Any] = {}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def step(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        print(f"{name} not on PATH")
        sys.exit(2)
    return found


# ---------------------------------------------------------------------- inputs
CLIP_SPECS: list[tuple[str, str]] = [
    ("ce_bars.mp4", f"smptebars=size=1280x720:rate=25:duration={SOURCE_S}"),
    ("ce_test.mp4", f"testsrc=size=1280x720:rate=25:duration={SOURCE_S}"),
    ("ce_rgb.mp4", f"rgbtestsrc=size=1280x720:rate=25:duration={SOURCE_S}"),
]
TRACK_NAME = "ce_music.wav"


def build_inputs() -> tuple[list[Path], Path]:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = binary("ffmpeg")

    clips: list[Path] = []
    for name, source in CLIP_SPECS:
        destination = CLIP_DIR / name
        if not destination.exists():
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", source,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
                    str(destination),
                ],
                check=True, capture_output=True, timeout=300,
            )
        clips.append(destination)

    track = CLIP_DIR / TRACK_NAME
    if not track.exists():
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=40",
                "-ac", "2", str(track),
            ],
            check=True, capture_output=True, timeout=300,
        )
    return clips, track


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


def wait_for_job(job_id: str, timeout_s: int = JOB_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = httpx.get(f"{API}/api/jobs/{job_id}", timeout=30).json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} stuck in {last.get('status')} after {timeout_s}s")


def wait_for_render(project_id: str, render_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = httpx.get(f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30).json()
        if last["status"] in {"ready", "failed", "cancelled"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"render {render_id} stuck in {last.get('status')}")


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [binary("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return json.loads(result.stdout)


def plan_url(project_id: str, suffix: str = "") -> str:
    return f"{API}/api/projects/{project_id}/edit-plan{suffix}"


def get_plan(project_id: str, plan_id: str) -> dict[str, Any]:
    response = httpx.get(plan_url(project_id, f"/{plan_id}"), timeout=30)
    response.raise_for_status()
    return response.json()["plan"]


def versions(project_id: str) -> dict[str, Any]:
    response = httpx.get(plan_url(project_id, "/versions"), timeout=30)
    response.raise_for_status()
    return response.json()


def current_version(project_id: str) -> dict[str, Any]:
    listing = versions(project_id)
    return next(item for item in listing["items"] if item["is_current"])


def co_edit(project_id: str, **body: Any) -> httpx.Response:
    return httpx.post(plan_url(project_id, "/co-edit"), json=body, timeout=120)


def preview(project_id: str, **body: Any) -> httpx.Response:
    return httpx.post(plan_url(project_id, "/co-edit/preview"), json=body, timeout=120)


def render_plan(project_id: str, plan_id: str, name: str) -> tuple[Path, dict[str, Any], int]:
    """Render one plan and download it. The unchanged Phase 4 pipeline."""
    queued = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan_id},
        timeout=60,
    )
    check(f"{name}: render queued", queued.status_code == 202, str(queued.status_code))
    render_id = queued.json()["id"]

    started = time.monotonic()
    render = wait_for_render(project_id, render_id)
    elapsed = int((time.monotonic() - started) * 1000)
    check(f"{name}: render succeeded", render["status"] == "ready", str(render.get("error")))

    detail = httpx.get(f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30).json()
    output = CLIP_DIR / f"phase10_{name}.mp4"
    url = detail.get("playback_url")
    check(f"{name}: a presigned playback URL is issued", bool(url))
    if url:
        output.write_bytes(httpx.get(url, timeout=300).content)
    return output, detail, elapsed


SUBTITLES = {
    "style": "clean",
    "position": "bottom",
    "cues": [
        {"start_ms": 200, "end_ms": 2_600, "text": "Phase 10: the AI co-editor"},
        {"start_ms": 3_000, "end_ms": 5_400, "text": "Xin chào - phụ đề tiếng Việt"},
        {"start_ms": 6_000, "end_ms": 8_400, "text": "Ask for a change, get a version"},
    ],
}


# ------------------------------------------------------------------- scenarios
def scenario_media() -> tuple[str, list[str], str]:
    step("1. USER MEDIA - import three clips and a music track")
    clips, track = build_inputs()
    project_id = create_project("Phase 10 AI Co-Editor")
    print(f" ... project {project_id}")

    media_ids: list[str] = []
    for path in clips:
        media_id, job = upload(project_id, path)
        media_ids.append(media_id)
        if job:
            result = wait_for_job(job)
            check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    track_id, job = upload(project_id, track)
    if job:
        result = wait_for_job(job)
        check(f"ingest {track.name}", result["status"] == "succeeded", result["status"])

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    ready = [m for m in listing["items"] if m["status"] == "ready"]
    check("all four assets ready", len(ready) == 4, f"{len(ready)}/4")
    return project_id, media_ids, track_id


def scenario_analysis(project_id: str) -> None:
    step("2. ANALYSIS - the unchanged CPU lane")
    accepted = httpx.post(
        f"{API}/api/projects/{project_id}/analysis", json={"lanes": ["cpu"]}, timeout=60
    )
    check("analysis accepted", accepted.status_code == 202, str(accepted.status_code))
    for job in accepted.json()["jobs"]:
        result = wait_for_job(job["id"])
        check(f"analysis {job['id'][:8]}", result["status"] == "succeeded", result["status"])


def scenario_initial_edit(project_id: str, track_id: str) -> dict[str, Any]:
    """The edit the user starts from: planned, then subtitled by hand."""
    step("3. INITIAL EDIT - planned, rendered, then subtitled on the timeline")

    planned = httpx.post(
        plan_url(project_id),
        json={
            "mode": "rules",
            "max_clips": 3,
            "min_clips": 2,
            "target_duration_ms": 12_000,
            "aspect_ratio": "16:9",
            "fps": 30,
            "quality": "draft",
            "music": {
                "media_id": track_id,
                "source_in_ms": 0,
                "source_out_ms": 12_000,
                "timeline_start_ms": 0,
                "volume": 0.7,
                "fade_in_ms": 0,
                "fade_out_ms": 1_500,
            },
        },
        timeout=120,
    )
    check("the planner produced an edit", planned.status_code == 201, planned.text[:300])
    if planned.status_code != 201:
        return {}

    first = versions(project_id)
    check("the planned edit is version 1", first["total"] == 1 and first["items"][0]["version"] == 1)
    check("it is recorded as generated", first["items"][0]["origin"] == "generated")
    check("there is nothing to undo yet", first["can_undo"] is False)

    # Render version 1: the acceptance's "CREATE INITIAL EDIT -> RENDER".
    output, _detail, elapsed = render_plan(project_id, planned.json()["id"], "v1")
    check("version 1 rendered to a file", output.exists() and output.stat().st_size > 0)
    _report["v1_render_ms"] = elapsed

    # The user adds subtitles on the timeline. A hand-cut edit joins the same
    # history, which is what lets the co-editor patch it next.
    document = planned.json()["plan"]
    manual = {
        "segments": [
            {
                "media_id": segment["media_id"],
                "source_in_ms": segment["source_in_ms"],
                "source_out_ms": segment["source_out_ms"],
                "transition_in": segment["transition_in"],
                "transition_ms": segment.get("transition_ms", 0),
                "effects": segment.get("effects", []),
            }
            for segment in document["segments"]
        ],
        "aspect_ratio": document["output"]["aspect_ratio"],
        "fps": document["output"]["fps"],
        "quality": "draft",
        "audio": document["output"]["audio"],
        "music": {
            "media_id": document["music"]["media_id"],
            "source_in_ms": document["music"]["source_in_ms"],
            "source_out_ms": document["music"]["source_out_ms"],
            "timeline_start_ms": document["music"]["timeline_start_ms"],
            "volume": document["music"]["gain"],
            "fade_in_ms": document["music"]["fade_in_ms"],
            "fade_out_ms": document["music"]["fade_out_ms"],
        },
        "subtitles": SUBTITLES,
        "derived_from_edit_plan_id": planned.json()["id"],
    }
    stored = httpx.post(plan_url(project_id, "/manual"), json=manual, timeout=60)
    check("the subtitled timeline was accepted", stored.status_code == 201, stored.text[:300])
    if stored.status_code != 201:
        return {}

    history = versions(project_id)
    check("the hand-cut edit is version 2", history["items"][0]["version"] == 2)
    check("it is recorded as manual", history["items"][0]["origin"] == "manual")
    check("undo is now possible", history["can_undo"] is True)

    _report["initial_segments"] = len(document["segments"])
    _report["initial_duration_ms"] = stored.json()["total_duration_ms"]
    return stored.json()


def scenario_gate(project_id: str, baseline: dict[str, Any]) -> None:
    """Changes the server must refuse, with the edit left untouched."""
    step("4. THE GATE - changes that must be refused, and change nothing")

    before = get_plan(project_id, baseline["id"])
    before_versions = versions(project_id)["total"]

    def refuse(label: str, **body: Any) -> None:
        response = co_edit(project_id, **body)
        check(f"refused: {label}", response.status_code == 422, f"HTTP {response.status_code}")

    refuse("a clip that does not exist", operations=[{"kind": "REMOVE_SEGMENT", "segment": 9}])
    refuse(
        "an operation kind that does not exist",
        operations=[{"kind": "RUN_FFMPEG", "args": "-i /etc/passwd"}],
    )
    refuse(
        "a music volume outside its range",
        operations=[{"kind": "CHANGE_MUSIC_VOLUME", "value": 99}],
    )
    refuse(
        "a speed outside what the effect allows",
        operations=[{"kind": "ADD_EFFECT", "segment": 0, "effect": "slow_motion", "amount": 0.01}],
    )
    refuse(
        "a trim past the end of the real source",
        operations=[
            {"kind": "TRIM_SEGMENT", "segment": 0, "source_in_ms": 0, "source_out_ms": 900_000}
        ],
    )
    refuse(
        "a crossfade on the first clip",
        operations=[{"kind": "CHANGE_TRANSITION", "segment": 0, "transition": "crossfade"}],
    )
    refuse(
        "a change where one operation of two is invalid",
        operations=[
            {"kind": "CHANGE_MUSIC_VOLUME", "value": 0.3},
            {"kind": "REMOVE_SEGMENT", "segment": 7},
        ],
    )

    # A geometry a client tried to smuggle in is not read at all: the operation
    # is accepted for the field it does have and the extra keys never existed.
    smuggled = preview(
        project_id,
        operations=[
            {"kind": "CHANGE_OUTPUT_PRESET", "quality": "draft", "width": 4096, "height": 2160}
        ],
    )
    check(
        "a smuggled width and height are ignored rather than honoured",
        smuggled.status_code == 200 and "4096" not in smuggled.text,
        f"HTTP {smuggled.status_code}",
    )

    after = get_plan(project_id, baseline["id"])
    check("the edit is byte-identical after every refusal", before == after)
    check("no version was created", versions(project_id)["total"] == before_versions)


def scenario_change(project_id: str, baseline: dict[str, Any]) -> dict[str, Any]:
    """The first change: "Make the opening faster and lower the music." """
    step('5. CHANGE - "Make the opening faster and lower the music to 40%."')

    request = "Make the opening faster and lower the music to 40%."

    shown = preview(project_id, request_text=request)
    check("the change previews", shown.status_code == 200, shown.text[:300])
    body = shown.json()
    check("the preview succeeded", body["ok"] is True, body.get("detail", ""))
    check(
        "it was resolved without a model",
        body["source"] == "rules",
        body["source"],
    )
    check(
        "the preview names both halves of the request",
        len(body["operations"]) == 2,
        str([op["kind"] for op in body["operations"]]),
    )
    fields = {entry["field"] for entry in body["diff"]["entries"]}
    check("the diff shows the music change", "music_gain" in fields, str(sorted(fields)))
    check(
        "the diff shows the opening changing length",
        "segment_duration" in fields or "total_duration" in fields,
        str(sorted(fields)),
    )

    # Nothing has been written yet.
    check("previewing wrote no version", versions(project_id)["total"] == 2)
    check(
        "previewing left the plan alone",
        get_plan(project_id, baseline["id"])["music"]["gain"] == 0.7,
    )

    applied = co_edit(project_id, request_text=request)
    check("the change applied", applied.status_code == 201, applied.text[:400])
    if applied.status_code != 201:
        return {}

    result = applied.json()
    document = result["plan"]["plan"]
    check("a new version was created", result["version"]["version"] == 3)
    check("it is recorded as a co-edit", result["version"]["origin"] == "co_edit")
    check("the music is at 40%", document["music"]["gain"] == 0.4, str(document["music"]["gain"]))

    opening = document["segments"][0]
    check(
        "the opening carries a speed change",
        any(effect["kind"] == "speed_up" for effect in opening["effects"]),
        str(opening["effects"]),
    )
    check(
        "the other clips were not touched",
        all(not segment["effects"] for segment in document["segments"][1:]),
    )
    check(
        "the same clips are still in the edit",
        [segment["media_id"] for segment in document["segments"]]
        == [segment["media_id"] for segment in get_plan(project_id, baseline["id"])["segments"]],
    )
    check(
        "the subtitles survived the change",
        document.get("subtitles") is not None
        and len(document["subtitles"]["cues"]) >= 1,
    )
    check(
        "the previous plan is untouched",
        get_plan(project_id, baseline["id"])["music"]["gain"] == 0.7,
    )

    output, _detail, elapsed = render_plan(project_id, result["plan"]["id"], "v3")
    check("version 3 rendered", output.exists() and output.stat().st_size > 0)
    _report["v3_render_ms"] = elapsed
    _report["v3_operations"] = result["diff"]["applied"]
    return result


def scenario_undo(project_id: str, baseline: dict[str, Any]) -> None:
    step('6. UNDO - "Undo that." restores version 2 exactly')

    before = get_plan(project_id, baseline["id"])

    response = httpx.post(plan_url(project_id, "/versions/undo"), timeout=30)
    check("undo accepted", response.status_code == 200, response.text[:300])
    restored = response.json()
    check("version 2 is current again", restored["version"] == 2, str(restored["version"]))
    check("it points at the original plan", restored["edit_plan_id"] == baseline["id"])

    after = get_plan(project_id, restored["edit_plan_id"])
    check("the restored plan is byte-identical, not recomputed", before == after)

    history = versions(project_id)
    check("the undone version is still in the history", history["total"] == 3)
    check("redo is now possible", history["can_redo"] is True)

    # Redo and undo again, to prove the head moves both ways.
    forward = httpx.post(plan_url(project_id, "/versions/redo"), timeout=30)
    check("redo accepted", forward.status_code == 200, forward.text[:200])
    check("version 3 is current again", forward.json()["version"] == 3)

    back = httpx.post(plan_url(project_id, "/versions/undo"), timeout=30)
    check("undo again", back.status_code == 200 and back.json()["version"] == 2)


def scenario_second_change(project_id: str) -> dict[str, Any]:
    step('7. CHANGE - "Remove the third clip and use bold subtitles."')

    before = current_version(project_id)
    before_plan = get_plan(project_id, before["edit_plan_id"])
    clips_before = len(before_plan["segments"])

    applied = co_edit(project_id, request_text="Remove the third clip and use bold subtitles.")
    check("the change applied", applied.status_code == 201, applied.text[:400])
    if applied.status_code != 201:
        return {}

    result = applied.json()
    document = result["plan"]["plan"]
    check(
        "a clip was removed",
        len(document["segments"]) == clips_before - 1,
        f"{len(document['segments'])} of {clips_before}",
    )
    check(
        "the remaining clips are the first two, in order",
        [segment["media_id"] for segment in document["segments"]]
        == [segment["media_id"] for segment in before_plan["segments"][:2]],
    )
    check("orders were renumbered", [s["order"] for s in document["segments"]] == [0, 1])
    check(
        "the subtitles are bold",
        document["subtitles"]["style"] == "bold",
        str(document["subtitles"]["style"]),
    )
    check(
        "the subtitle text was not rewritten",
        document["subtitles"]["cues"][0]["text"] == SUBTITLES["cues"][0]["text"],
    )
    check(
        "the branch created a new version",
        result["version"]["version"] >= 4,
        str(result["version"]["version"]),
    )
    check(
        "the applied changes are listed for the user",
        len(result["diff"]["applied"]) == 2,
        str(result["diff"]["applied"]),
    )

    _report["final_operations"] = result["diff"]["applied"]
    _report["final_plan_duration_ms"] = document["total_duration_ms"]
    return result


def scenario_llm(project_id: str) -> None:
    """The model path, when this deployment has one."""
    step("8. THE MODEL - exercised when a provider is configured")

    capabilities = httpx.get(f"{API}/api/planner/capabilities", timeout=30).json()
    available = bool(capabilities.get("ai_available"))
    provider = capabilities.get("provider")
    is_stub = bool(capabilities.get("is_stub"))
    _report["ai_available"] = available
    _report["ai_provider"] = provider
    _report["ai_is_stub"] = is_stub

    check(
        "the operation vocabulary is declared to clients",
        len(capabilities.get("operations", [])) == 16,
        str(len(capabilities.get("operations", []))),
    )

    before = current_version(project_id)
    creative = "Give the whole thing a more cinematic feel."

    if not available:
        print(" ... no AI provider configured; checking the refusal path instead")
        response = co_edit(project_id, request_text=creative)
        check(
            "a request needing interpretation is refused, not guessed at",
            response.status_code == 422,
            f"HTTP {response.status_code}",
        )
        check(
            "the refusal explains what the rules can do",
            "remove clip" in response.text or "40%" in response.text,
        )
        check("the edit is unchanged", current_version(project_id)["id"] == before["id"])
        return

    label = "stub (not a model)" if is_stub else f"{provider}"
    print(f" ... provider: {label}")
    shown = preview(project_id, request_text=creative)
    check("the model was asked", shown.status_code == 200, shown.text[:300])
    body = shown.json()
    check("the proposal came from the model path", body["source"] == "llm", body["source"])
    check("a model proposal is previewed before it lands", body["needs_confirmation"] is True)

    if not body["ok"]:
        # A model that could not express the request is a legitimate outcome and
        # must leave the edit alone.
        check(
            "a model that could not answer left the edit alone",
            current_version(project_id)["id"] == before["id"],
        )
        check("the failure is named", bool(body["failure"]), str(body.get("detail")))
        return

    check(
        "every proposed operation is in the closed vocabulary",
        all(
            op["kind"] in {item["kind"] for item in capabilities["operations"]}
            for op in body["operations"]
        ),
        str([op["kind"] for op in body["operations"]]),
    )
    check(
        "the proposal carries no path, filter or command",
        not any(
            key in json.dumps(body["operations"])
            for key in ("/", "drawtext", "ffmpeg", "http", "\\")
        ),
    )

    applied = co_edit(
        project_id,
        operations=body["operations"],
        base_version_id=body["base_version_id"],
    )
    check(
        "the confirmed proposal applied",
        applied.status_code == 201,
        applied.text[:300],
    )
    if applied.status_code == 201:
        _report["llm_operations"] = applied.json()["diff"]["applied"]


def scenario_final_render(project_id: str) -> Path:
    step("9. RENDER - the current version, through the unchanged pipeline")

    current = current_version(project_id)
    output, detail, elapsed = render_plan(project_id, current["edit_plan_id"], "final")
    _report["final_render_ms"] = elapsed
    _report["final_output_bytes"] = output.stat().st_size if output.exists() else 0
    _report["render_metrics"] = detail.get("metrics")

    # The version listing knows its plan was rendered, which is what the history
    # panel shows.
    after = current_version(project_id)
    check(
        "the version records its render",
        after["render_status"] == "ready",
        str(after.get("render_status")),
    )
    return output


def scenario_verify(project_id: str, output: Path) -> None:
    step("10. MP4 - verified independently with ffprobe")

    if not output.exists():
        check("the file exists", False)
        return

    probe = ffprobe(output)
    streams = probe["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)

    check("a video stream exists", video is not None)
    check("an audio stream exists", audio is not None)
    if video is None:
        return

    check("container is mp4", "mp4" in probe["format"]["format_name"],
          probe["format"]["format_name"])
    check("video codec is h264", video["codec_name"] == "h264", video["codec_name"])
    if audio:
        check("audio codec is aac", audio["codec_name"] == "aac", audio["codec_name"])
    check("resolution is 1280x720", (video["width"], video["height"]) == (1280, 720),
          f"{video['width']}x{video['height']}")

    numerator, _, denominator = video.get("avg_frame_rate", "0/1").partition("/")
    fps = float(numerator) / float(denominator or 1)
    check("frame rate is 30", abs(fps - 30.0) < 0.01, f"{fps:.2f}")

    # The plan's own arithmetic, against the file that was produced from it.
    current = current_version(project_id)
    document = get_plan(project_id, current["edit_plan_id"])
    expected = document["total_duration_ms"]
    measured = int(float(probe["format"]["duration"]) * 1000)
    check(
        "the rendered duration matches the patched plan's arithmetic",
        abs(measured - expected) <= DURATION_TOLERANCE_MS,
        f"{measured} ms vs {expected} ms expected",
    )

    decode = subprocess.run(
        [binary("ffmpeg"), "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        capture_output=True, timeout=900,
    )
    check("the file decodes completely", decode.returncode == 0)
    check("decoding reported no errors", not decode.stderr.strip(),
          decode.stderr.decode(errors="replace")[:200])

    _report["output_duration_ms"] = measured
    _report["output_dimensions"] = f"{video['width']}x{video['height']}"
    _report["output_fps"] = round(fps, 2)


def scenario_history(project_id: str) -> None:
    step("11. HISTORY - what the panel reads back")

    history = versions(project_id)
    check("every change is a version", history["total"] >= 4, str(history["total"]))
    check("exactly one version is current", sum(1 for v in history["items"] if v["is_current"]) == 1)
    check(
        "each co-edit says what it did",
        all(v["applied"] for v in history["items"] if v["origin"] == "co_edit"),
    )
    check(
        "the request text was never stored",
        "lower the music" not in json.dumps(history).lower(),
    )
    check(
        "a digest of the request was",
        any(v["request_digest"] for v in history["items"] if v["origin"] == "co_edit"),
    )

    _report["versions"] = [
        {"version": v["version"], "origin": v["origin"], "applied": v["applied"]}
        for v in reversed(history["items"])
    ]


def main() -> int:
    print("VisionForge - Phase 10 acceptance (AI co-editor, versions, undo)")
    print("=" * 64)

    project_id, _media_ids, track_id = scenario_media()
    scenario_analysis(project_id)

    baseline = scenario_initial_edit(project_id, track_id)
    if not baseline:
        print("\nthe initial edit could not be built; the rest cannot be checked")
        return 1

    scenario_gate(project_id, baseline)

    changed = scenario_change(project_id, baseline)
    if not changed:
        print("\nthe first change was refused; the rest cannot be checked")
        return 1

    scenario_undo(project_id, baseline)
    scenario_second_change(project_id)
    scenario_llm(project_id)

    output = scenario_final_render(project_id)
    scenario_verify(project_id, output)
    scenario_history(project_id)

    _report["ffmpeg"] = subprocess.run(
        [binary("ffmpeg"), "-version"], capture_output=True, text=True, timeout=60
    ).stdout.splitlines()[0]

    step("Summary")
    print(json.dumps(_report, indent=2, sort_keys=True, ensure_ascii=False))

    print("\n" + "=" * 64)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All Phase 10 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
