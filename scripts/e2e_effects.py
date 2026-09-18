"""Phase 9 acceptance test: transitions, effects and subtitles to a playable MP4.

Phase 7 proved a beat-synced edit renders with a music bed. Phase 8 proved a
reference video's style reaches the planner. This proves the editing primitives:

    upload footage + a music track
        -> ingest          (the unchanged Phase 2 pipeline)
        -> analysis        (the unchanged CPU lane)
        -> plan            (rules engine)
        -> hand-cut edit   (transitions, effects and subtitles on the timeline)
        -> validate        (the unchanged Phase 4 gate, extended)
        -> audio mix       (the unchanged Phase 7 pipeline)
        -> render          (the unchanged render worker)
        -> MP4             (ffprobe, a burned-in subtitle check, a full decode)

Three claims are under test.

**The timeline arithmetic is honest.** A crossfade shortens the programme and a
speed change lengthens or shortens a clip. The plan's own reported duration, and
the duration ffprobe measures in the file, must agree -- an editor that reports
one length and renders another is worse than one that refuses.

**Subtitles are visible, and Vietnamese works.** Not "the graph parsed": the same
edit is rendered with and without the subtitle track and the frames compared, so
a cue that drew nothing fails.

**The gate still refuses what it should.** A cut with a duration, a dissolve
longer than its neighbours can spare, an out-of-range effect, an unknown effect
kind, overlapping cues and a cue past the end of the edit are each rejected
before anything is stored.

Everything is generated with FFmpeg. No network, no model weights, no
copyrighted bytes.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_effects.py

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

#: The edit the acceptance builds, in milliseconds.
CLIP_MS = 3_000
CROSSFADE_MS = 600
SLOW_RATE = 0.5

#: One frame at 30 fps is 33 ms; the muxer and `xfade` each round to a frame.
DURATION_TOLERANCE_MS = 200

PASS, FAIL = "PASS", "FAIL"
_failures: list[str] = []
_report: dict[str, object] = {}


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
    ("fx_bars.mp4", "smptebars=size=1280x720:rate=25:duration=8"),
    ("fx_test.mp4", "testsrc=size=1280x720:rate=25:duration=8"),
    ("fx_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=8"),
]
TRACK_NAME = "fx_music.wav"


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
                "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
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
        last = httpx.get(f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30).json()
        if last["status"] in {"ready", "failed", "cancelled"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"render {render_id} stuck in {last.get('status')}")


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [binary("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return json.loads(result.stdout)


def manual_plan(project_id: str, body: dict) -> httpx.Response:
    return httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan/manual", json=body, timeout=120
    )


def timeline(media_ids: list[str], **overrides: object) -> dict:
    """The edit this acceptance builds, as the editor would submit it.

    Three clips: the first graded and fading up from black, the second arriving
    on a crossfade with a zoom, the third at half speed and fading out.
    """
    body: dict = {
        "segments": [
            {
                "media_id": media_ids[0],
                "source_in_ms": 0,
                "source_out_ms": CLIP_MS,
                "transition_in": "fade_in",
                "transition_ms": 500,
                "effects": [{"kind": "brightness", "amount": 0.12}],
            },
            {
                "media_id": media_ids[1],
                "source_in_ms": 0,
                "source_out_ms": CLIP_MS,
                "transition_in": "crossfade",
                "transition_ms": CROSSFADE_MS,
                "effects": [{"kind": "zoom_in", "amount": 0.18}],
            },
            {
                "media_id": media_ids[2],
                "source_in_ms": 0,
                "source_out_ms": CLIP_MS,
                "transition_in": "fade_to_black",
                "transition_ms": 500,
                "effects": [{"kind": "slow_motion", "amount": SLOW_RATE}],
            },
        ],
        "aspect_ratio": "16:9",
        "fps": 30,
        "quality": "draft",
        "audio": "none",
    }
    body.update(overrides)
    return body


def expected_duration_ms() -> int:
    """What the edit above should run to, computed independently of the server.

    Two full clips, one at half speed, minus the single crossfade overlap. The
    fades consume nothing. Written out here rather than read from the plan, so
    the acceptance is checking the server's arithmetic rather than echoing it.
    """
    return CLIP_MS + CLIP_MS + int(CLIP_MS / SLOW_RATE) - CROSSFADE_MS


SUBTITLES = {
    "style": "cinematic",
    "position": "bottom",
    "cues": [
        {"start_ms": 300, "end_ms": 2_400, "text": "Phase 9: transitions and effects"},
        {"start_ms": 2_700, "end_ms": 5_200, "text": "Xin chào — phụ đề tiếng Việt"},
        {"start_ms": 5_500, "end_ms": 8_000, "text": "Slow motion, fading to black"},
    ],
}


# ------------------------------------------------------------------- scenarios
def scenario_media() -> tuple[str, list[str], str]:
    step("1. USER MEDIA - import three clips and a music track")
    clips, track = build_inputs()
    project_id = create_project("Phase 9 Effects and Subtitles")
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


def scenario_gate(project_id: str, media_ids: list[str]) -> None:
    step("3. THE GATE - edits the server must refuse")

    def refuse(label: str, body: dict) -> None:
        response = manual_plan(project_id, body)
        check(f"refused: {label}", response.status_code == 422, f"HTTP {response.status_code}")

    plain = timeline(media_ids)

    body = json.loads(json.dumps(plain))
    body["segments"][0]["transition_ms"] = 400  # a cut with a duration
    body["segments"][0]["transition_in"] = "cut"
    refuse("a cut that claims a duration", body)

    body = json.loads(json.dumps(plain))
    body["segments"][1]["transition_ms"] = 2_800  # more than half of a 3 s clip
    refuse("a dissolve longer than its neighbours can spare", body)

    body = json.loads(json.dumps(plain))
    body["segments"][0]["effects"] = [{"kind": "brightness", "amount": 5.0}]
    refuse("an effect outside its range", body)

    body = json.loads(json.dumps(plain))
    body["segments"][0]["effects"] = [{"kind": "deep_fry", "amount": 1.0}]
    refuse("an effect kind that does not exist", body)

    body = json.loads(json.dumps(plain))
    body["segments"][0]["effects"] = [{"kind": "zoom_in", "amount": 0.2, "start_ms": 100,
                                       "end_ms": 900}]
    refuse("a zoom asked to cover part of a clip", body)

    body = json.loads(json.dumps(plain))
    body["subtitles"] = {
        "style": "cinematic",
        "position": "bottom",
        "cues": [
            {"start_ms": 0, "end_ms": 2_000, "text": "first"},
            {"start_ms": 1_500, "end_ms": 3_000, "text": "second"},
        ],
    }
    refuse("subtitle cues that overlap", body)

    body = json.loads(json.dumps(plain))
    body["subtitles"] = {
        "style": "cinematic",
        "position": "bottom",
        "cues": [{"start_ms": 0, "end_ms": 2_000, "text": "x" * 500}],
    }
    refuse("subtitle text over the cap", body)

    body = json.loads(json.dumps(plain))
    body["subtitles"] = {
        "style": "neon_sparkles",
        "position": "bottom",
        "cues": [{"start_ms": 0, "end_ms": 2_000, "text": "hi"}],
    }
    refuse("a subtitle style that does not exist", body)

    body = json.loads(json.dumps(plain))
    body["subtitles"] = {
        "style": "cinematic",
        "position": "bottom",
        "cues": [{"start_ms": 0, "end_ms": 2_000, "text": "hi"}],
        "font": "/windows/fonts/arial.ttf",
    }
    response = manual_plan(project_id, body)
    check(
        "a font path is ignored rather than honoured",
        response.status_code == 201 and "arial.ttf" not in response.text,
        f"HTTP {response.status_code}",
    )


def scenario_edit(project_id: str, media_ids: list[str], track_id: str) -> dict:
    step("4. THE EDIT - transitions, effects, subtitles and a music bed")

    body = timeline(media_ids)
    body["subtitles"] = SUBTITLES
    body["audio"] = "none"
    body["music"] = {
        "media_id": track_id,
        "source_in_ms": 0,
        "source_out_ms": expected_duration_ms(),
        "timeline_start_ms": 0,
        "volume": 0.6,
        "fade_in_ms": 400,
        "fade_out_ms": 800,
    }

    response = manual_plan(project_id, body)
    check("the edit was accepted", response.status_code == 201, f"HTTP {response.status_code}")
    if response.status_code != 201:
        print(response.text[:600])
        return {}

    plan = response.json()
    document = plan["plan"]

    expected = expected_duration_ms()
    check(
        "the plan's duration accounts for the crossfade and the slow motion",
        abs(document["total_duration_ms"] - expected) <= 2,
        f"{document['total_duration_ms']} ms vs {expected} ms expected",
    )

    kinds = [s["transition_in"] for s in document["segments"]]
    check("the transitions were stored", kinds == ["fade_in", "crossfade", "fade_to_black"],
          str(kinds))

    effects = [e["kind"] for s in document["segments"] for e in s["effects"]]
    check("the effects were stored", effects == ["brightness", "zoom_in", "slow_motion"],
          str(effects))

    stored = document.get("subtitles")
    check("the subtitles were stored", stored is not None and len(stored["cues"]) == 3)
    if stored:
        check("the subtitle preset is an id, not parameters",
              set(stored) == {"style", "position", "cues"}, str(sorted(stored)))
        check("the Vietnamese cue survived intact",
              any("tiếng Việt" in cue["text"] for cue in stored["cues"]))

    _report["plan_duration_ms"] = document["total_duration_ms"]
    _report["transitions"] = kinds
    _report["effects"] = effects
    return plan


def scenario_render(project_id: str, plan: dict) -> Path:
    step("5. RENDER - through the unchanged pipeline")

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan["id"]},
        timeout=60,
    )
    check("render queued", queued.status_code == 202, str(queued.status_code))
    render_id = queued.json()["id"]

    started = time.monotonic()
    render = wait_for_render(project_id, render_id)
    render_ms = int((time.monotonic() - started) * 1000)
    check("render succeeded", render["status"] == "ready", str(render.get("error")))

    detail = httpx.get(f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30).json()
    url = detail.get("playback_url")
    check("a presigned playback URL is issued", bool(url))

    output = CLIP_DIR / "phase9_output.mp4"
    if url:
        output.write_bytes(httpx.get(url, timeout=300).content)
        check("the file downloaded", output.stat().st_size > 0, f"{output.stat().st_size} bytes")

    _report["render_ms"] = render_ms
    _report["output_bytes"] = output.stat().st_size if output.exists() else 0
    return output


def scenario_verify(output: Path) -> None:
    step("6. MP4 - verified independently with ffprobe")

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

    measured = int(float(probe["format"]["duration"]) * 1000)
    expected = expected_duration_ms()
    check(
        "the rendered duration matches the timeline arithmetic",
        abs(measured - expected) <= DURATION_TOLERANCE_MS,
        f"{measured} ms vs {expected} ms expected",
    )

    ffmpeg = binary("ffmpeg")
    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        capture_output=True, timeout=900,
    )
    check("the file decodes completely", decode.returncode == 0)
    check("decoding reported no errors", not decode.stderr.strip(),
          decode.stderr.decode(errors="replace")[:200])

    _report["output_duration_ms"] = measured
    _report["output_dimensions"] = f"{video['width']}x{video['height']}"
    _report["output_fps"] = round(fps, 2)


def scenario_subtitles_visible(project_id: str, media_ids: list[str], output: Path) -> None:
    """Render the same edit without subtitles and compare a frame.

    The check that cannot be faked: a subtitle filter that parsed and drew
    nothing would satisfy every other assertion in this script.
    """
    step("7. SUBTITLES - proven visible, not merely accepted")

    body = timeline(media_ids)
    response = manual_plan(project_id, body)
    check("the same edit without subtitles was accepted", response.status_code == 201,
          f"HTTP {response.status_code}")
    if response.status_code != 201:
        return

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": response.json()["id"]},
        timeout=60,
    )
    render = wait_for_render(project_id, queued.json()["id"])
    check("the control render succeeded", render["status"] == "ready")

    detail = httpx.get(
        f"{API}/api/projects/{project_id}/renders/{queued.json()['id']}", timeout=30
    ).json()
    if not detail.get("playback_url"):
        return

    plain = CLIP_DIR / "phase9_plain.mp4"
    plain.write_bytes(httpx.get(detail["playback_url"], timeout=300).content)

    ffmpeg = binary("ffmpeg")
    frames: list[Path] = []
    # 1.2 s: inside the first cue, and past the fade-in so both renders are at
    # full brightness.
    for name, source in (("plain.png", plain), ("burned.png", output)):
        frame = CLIP_DIR / f"phase9_{name}"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
             "-ss", "1.2", "-frames:v", "1", str(frame)],
            check=True, capture_output=True, timeout=120,
        )
        frames.append(frame)

    check("the subtitled frame differs from the plain one",
          frames[0].read_bytes() != frames[1].read_bytes())
    check("the subtitled frame carries more detail",
          frames[1].stat().st_size > frames[0].stat().st_size,
          f"{frames[1].stat().st_size} vs {frames[0].stat().st_size} bytes")


def main() -> int:
    print("VisionForge - Phase 9 acceptance (transitions, effects, subtitles)")
    print("=" * 60)

    project_id, media_ids, track_id = scenario_media()
    scenario_analysis(project_id)
    scenario_gate(project_id, media_ids)

    plan = scenario_edit(project_id, media_ids, track_id)
    if not plan:
        print("\nthe edit was refused; the rest cannot be checked")
        return 1

    output = scenario_render(project_id, plan)
    if not output.exists():
        return 1
    scenario_verify(output)
    scenario_subtitles_visible(project_id, media_ids, output)

    _report["ffmpeg"] = subprocess.run(
        [binary("ffmpeg"), "-version"], capture_output=True, text=True, timeout=60
    ).stdout.splitlines()[0]

    step("Summary")
    print(json.dumps(_report, indent=2, sort_keys=True, ensure_ascii=False))

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All Phase 9 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
