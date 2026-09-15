"""Phase 7 acceptance test: a scored edit, from footage and a track to an MP4.

Phase 4 proved a planned edit renders. Phase 6 proved a hand-cut one does. This
proves a **scored** one does, and that every audio decision the user makes
survives the whole pipeline:

    upload 4 clips + 1 generated music track
        -> ingest        (the unchanged Phase 2 pipeline; audio needs no proxy)
        -> analysis      (CPU lane, now including beat detection on the track)
        -> selection     (the unchanged Phase 4 scorer)
        -> plan          (rules engine, with the cue and the beat grid)
        -> beat sync     (clip length quantised to whole beats)
        -> audio mix     (source + music, independent levels, fades)
        -> render        (the unchanged render worker)
        -> MP4           (verified with ffprobe and a full decode pass)

The negative cases matter as much as the positive one. A cue that names another
project's track, one trimmed past the end of its file, one with impossible fades
and one naming a video must each be refused *before* anything is stored, with a
reason the editor can show. A bed that could smuggle any of those past the gate
would mean the audio path had acquired a weaker guarantee than the video path,
which is the one thing Phase 7 was not allowed to do.

**The music is generated, never downloaded.** It is a synthesised metronome at a
known tempo: no network, no provider, no copyrighted bytes in the repository or
on the wire.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_music.py

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

#: The tempo the generated track is built at. Detection has to find this.
TRACK_BPM = 120.0
TRACK_SECONDS = 90
#: Half a beat. Detection resolves to one STFT hop (~23 ms), so anything inside
#: this is agreement rather than luck.
BPM_TOLERANCE = 3.0

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
CLIP_SPECS: list[tuple[str, str]] = [
    ("music_bars.mp4", "smptebars=size=1280x720:rate=25:duration=8"),
    ("music_test.mp4", "testsrc=size=1280x720:rate=25:duration=8"),
    ("music_test2.mp4", "testsrc2=size=1280x720:rate=25:duration=8"),
    ("music_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=8"),
]

#: A metronome, synthesised by FFmpeg. A decaying sine burst on every beat --
#: broadband enough for spectral flux to see, and at a tempo this file *defines*
#: rather than merely contains, so "detected 119.9" is a measurable error.
TRACK_EXPR = (
    f"aevalsrc='0.6*sin(3000*2*PI*t)*exp(-24*mod(t,{60.0 / TRACK_BPM:.6f}))'"
    f":d={TRACK_SECONDS}:s=44100"
)


def ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        print("ffmpeg not on PATH")
        sys.exit(2)
    return binary


def build_inputs() -> tuple[list[Path], Path]:
    """Generate the clips and the track. Nothing is downloaded, nothing committed."""
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    binary = ffmpeg()

    clips: list[Path] = []
    for name, source in CLIP_SPECS:
        destination = CLIP_DIR / name
        if not destination.exists():
            subprocess.run(
                [
                    binary, "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", source,
                    # Silent video on purpose: the bed is the only sound, which
                    # is the case a highlight reel actually has.
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
                    str(destination),
                ],
                check=True, capture_output=True, timeout=300,
            )
        clips.append(destination)

    track = CLIP_DIR / f"music_{int(TRACK_BPM)}bpm.wav"
    if not track.exists():
        subprocess.run(
            [
                binary, "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", TRACK_EXPR, "-ac", "2", str(track),
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
        last = httpx.get(
            f"{API}/api/projects/{project_id}/renders/{render_id}", timeout=30
        ).json()
        if last["status"] in {"ready", "failed", "cancelled"}:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"render {render_id} stuck in {last.get('status')}")


def ffprobe(path: Path) -> dict:
    binary = shutil.which("ffprobe")
    assert binary
    result = subprocess.run(
        [
            binary, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return dict(json.loads(result.stdout))


def stream(info: dict, kind: str) -> dict | None:
    for candidate in info.get("streams", []):
        if candidate.get("codec_type") == kind:
            return dict(candidate)
    return None


def ffmpeg_version() -> str:
    result = subprocess.run(
        [ffmpeg(), "-version"], check=True, capture_output=True, text=True, timeout=30
    )
    return result.stdout.splitlines()[0]


# ------------------------------------------------------------------- scenarios
def scenario_prepare(clips: list[Path], track: Path) -> tuple[str, dict[str, str]]:
    step("1. MEDIA - import four clips and one generated track")
    project_id = create_project("Phase 7 Music and Beat Sync")
    print(f"{INFO} project {project_id}")

    media: dict[str, str] = {}
    for path in [*clips, track]:
        media_id, job_id = upload(project_id, path)
        media[path.name] = media_id
        assert job_id
        result = wait_for_job(job_id)
        check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    by_id = {item["id"]: item for item in listing["items"]}

    audio = [item for item in listing["items"] if item["kind"] == "audio"]
    check("the track ingested as audio", len(audio) == 1, f"{len(audio)} audio asset(s)")
    if audio:
        probed = audio[0]
        check(
            "ffprobe measured the track",
            probed["duration_ms"] is not None and probed["duration_ms"] > 0,
            f"{probed['duration_ms']} ms",
        )
        check("the track has channels", bool(probed["channels"]), str(probed["channels"]))
        # Audio gets neither a thumbnail nor a proxy; the ingest pipeline is
        # unchanged and simply skips both.
        check("audio needed no proxy", probed["has_proxy"] is False)

    video = [item for item in listing["items"] if item["kind"] == "video"]
    check("four clips ready", len(video) == 4, f"{len(video)}")
    _report["input_clips"] = len(video)
    _report["track_duration_ms"] = audio[0]["duration_ms"] if audio else None
    return project_id, {name: mid for name, mid in media.items() if mid in by_id}


def scenario_analysis(project_id: str, track_id: str) -> dict:
    step("2. ANALYSIS - the CPU lane, including beat detection")
    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis", json={"lanes": ["cpu"]}, timeout=60
    )
    check("analysis accepted", queued.status_code in (200, 202), str(queued.status_code))

    for job in queued.json().get("jobs", []):
        result = wait_for_job(job["id"])
        check(f"analysis job {job['id'][:8]}", result["status"] == "succeeded", result["status"])

    analysis = httpx.get(
        f"{API}/api/projects/{project_id}/media/{track_id}/analysis", timeout=30
    ).json()
    beats = next((r for r in analysis["items"] if r["analyzer"] == "beats"), None)
    check("the track has a beats record", beats is not None)
    if beats is None:
        return {}

    check("beat detection succeeded", beats["status"] == "ok", beats["status"])
    payload = beats["payload"]
    bpm = float(payload.get("bpm", 0))
    confidence = float(payload.get("confidence", 0))

    check(
        f"tempo is {TRACK_BPM:g} BPM",
        abs(bpm - TRACK_BPM) <= BPM_TOLERANCE,
        f"detected {bpm:.2f}",
    )
    check("confident enough to cut to", confidence >= 0.35, f"{confidence:.3f}")
    check("beats were found", int(payload.get("beat_count", 0)) > 20, str(payload.get("beat_count")))

    # Determinism: the same bytes must give the same answer, or a beat-synced
    # edit is not reproducible.
    again = httpx.get(
        f"{API}/api/projects/{project_id}/media/{track_id}/analysis", timeout=30
    ).json()
    repeat = next(r for r in again["items"] if r["analyzer"] == "beats")
    check("the stored grid is stable", repeat["payload"]["beats_ms"] == payload["beats_ms"])

    _report["bpm"] = round(bpm, 2)
    _report["beat_confidence"] = round(confidence, 4)
    _report["beat_count"] = payload.get("beat_count")
    return dict(payload)


def scenario_plan(project_id: str, track_id: str, beats: dict) -> dict:
    step("3. SELECTION + PLAN + BEAT SYNC - a scored, beat-matched cut")
    target_ms = 20_000
    response = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={
            "mode": "rules",
            "target_duration_ms": target_ms,
            "max_clips": 4,
            "min_clips": 2,
            "aspect_ratio": "16:9",
            "fps": 30,
            "quality": "draft",
            "beat_sync": True,
            "music": {
                "media_id": track_id,
                "source_in_ms": 0,
                "source_out_ms": 30_000,
                "timeline_start_ms": 0,
                "volume": 0.7,
                "fade_in_ms": 500,
                "fade_out_ms": 1_500,
            },
        },
        timeout=120,
    )
    check("plan created", response.status_code == 201, response.text[:200])
    if response.status_code != 201:
        sys.exit(1)

    plan = response.json()
    document = plan["plan"]

    check("the plan carries a music cue", document.get("music") is not None)
    cue = document.get("music") or {}
    check("the cue names the track", cue.get("media_id") == track_id)
    check("the cue kept the chosen level", abs(float(cue.get("gain", 0)) - 0.7) < 1e-6,
          str(cue.get("gain")))
    check("the cue kept its fades", (cue.get("fade_in_ms"), cue.get("fade_out_ms")) == (500, 1500),
          f"{cue.get('fade_in_ms')}/{cue.get('fade_out_ms')}")

    sync = document["metadata"].get("beat_sync", {})
    check("beat sync applied", sync.get("applied") is True, json.dumps(sync)[:160])
    beats_per_clip = sync.get("beats_per_clip")
    check(
        "beats per clip is a whole number",
        isinstance(beats_per_clip, int),
        f"{beats_per_clip!r} ({type(beats_per_clip).__name__})",
    )

    # The claim, checked on the finished plan: every cut lands on a beat.
    #
    # Against the period the plan *publishes*, not one recomputed from its BPM.
    # The reported BPM is rounded to two decimals for readability, and at 120 BPM
    # that is a 0.02 ms error per beat -- invisible on one cut and 0.8 ms after
    # forty, which is the tolerance itself. The planner works in the exact
    # period, so the check has to as well.
    period_ms = float(sync.get("beat_period_ms") or 60_000.0 / float(sync.get("bpm", TRACK_BPM)))
    cursor = 0
    worst = 0.0
    for segment in sorted(document["segments"], key=lambda s: s["order"]):
        cursor += segment["duration_ms"]
        remainder = cursor % period_ms
        worst = max(worst, min(remainder, period_ms - remainder))
    check("every cut lands on a beat", worst <= 1.0, f"worst {worst:.3f} ms off")

    check("clips were selected", len(document["segments"]) >= 2, str(len(document["segments"])))
    _report["selected_clips"] = len(document["segments"])
    _report["beats_per_clip"] = beats_per_clip
    _report["plan_duration_ms"] = document["total_duration_ms"]
    _report["music_cue_ms"] = cue.get("duration_ms")
    return dict(plan)


def scenario_rejections(project_id: str, track_id: str, clip_id: str) -> None:
    step("4. The gate - cues that must be refused before anything is stored")

    def manual(music: dict | None, label: str, expect: int = 422) -> None:
        body = {
            "segments": [
                {"media_id": clip_id, "source_in_ms": 0, "source_out_ms": 4_000},
                {"media_id": clip_id, "source_in_ms": 4_000, "source_out_ms": 8_000},
            ],
            "aspect_ratio": "16:9",
            "fps": 30,
            "quality": "draft",
        }
        if music is not None:
            body["music"] = music
        response = httpx.post(
            f"{API}/api/projects/{project_id}/edit-plan/manual", json=body, timeout=60
        )
        check(label, response.status_code == expect, f"HTTP {response.status_code}")

    manual(
        {"media_id": track_id, "source_in_ms": 0, "source_out_ms": 400_000},
        "a trim past the end of the track is refused",
    )
    manual(
        {"media_id": clip_id, "source_in_ms": 0, "source_out_ms": 5_000},
        "a video used as a music bed is refused",
    )
    manual(
        {
            "media_id": track_id, "source_in_ms": 0, "source_out_ms": 4_000,
            "fade_in_ms": 3_000, "fade_out_ms": 3_000,
        },
        "fades longer than the cue are refused",
    )
    manual(
        {"media_id": track_id, "source_in_ms": 0, "source_out_ms": 5_000, "volume": 9.0},
        "a level outside the range is refused",
    )
    manual(
        {"media_id": track_id, "source_in_ms": 9_000, "source_out_ms": 1_000},
        "an inverted range is refused",
    )

    # Another project's track. The row exists, but not here.
    other = create_project("Phase 7 Other Project")
    _clips, other_track = build_inputs()
    other_id, other_job = upload(other, other_track)
    assert other_job
    wait_for_job(other_job)
    manual(
        {"media_id": other_id, "source_in_ms": 0, "source_out_ms": 5_000},
        "another project's track is refused",
    )

    # And the control: the same shape, valid, is accepted.
    manual(
        {"media_id": track_id, "source_in_ms": 0, "source_out_ms": 8_000, "volume": 0.5},
        "a well-formed cue is accepted",
        expect=201,
    )


def scenario_render(project_id: str, plan: dict) -> dict:
    step("5. AUDIO MIX + RENDER")
    response = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan["id"]},
        timeout=60,
    )
    check("render queued", response.status_code == 202, str(response.status_code))
    render_id = response.json()["id"]

    started = time.monotonic()
    render = wait_for_render(project_id, render_id)
    elapsed = time.monotonic() - started
    check("render succeeded", render["status"] == "ready", json.dumps(render.get("error"))[:200])

    spec = render.get("spec") or {}
    check("the spec records a music mix", spec.get("music") is not None, json.dumps(spec)[:200])
    check("the spec records an audio codec", spec.get("audio_codec") == "aac", str(spec.get("audio_codec")))

    _report["render_ms"] = round(elapsed * 1000)
    _report["output_bytes"] = render.get("bytes_size")
    return dict(render)


def scenario_verify(project_id: str, render: dict, plan: dict) -> None:
    step("6. MP4 - verified independently with ffprobe")
    detail = httpx.get(
        f"{API}/api/projects/{project_id}/renders/{render['id']}", timeout=30
    ).json()
    url = detail.get("playback_url")
    check("a presigned playback URL is issued", bool(url))
    if not url:
        return

    output = CLIP_DIR / "phase7_output.mp4"
    output.write_bytes(httpx.get(url, timeout=300).content)
    check("the file downloaded", output.stat().st_size > 0, f"{output.stat().st_size} bytes")

    info = ffprobe(output)
    fmt = info.get("format", {})
    video = stream(info, "video")
    audio = stream(info, "audio")

    check("container is mp4", "mp4" in fmt.get("format_name", ""), fmt.get("format_name", ""))
    check("a video stream exists", video is not None)
    check("an audio stream exists", audio is not None)
    if video is None or audio is None:
        return

    check("video codec is h264", video.get("codec_name") == "h264", str(video.get("codec_name")))
    check("audio codec is aac", audio.get("codec_name") == "aac", str(audio.get("codec_name")))
    check(
        "resolution is 1280x720",
        (video.get("width"), video.get("height")) == (1280, 720),
        f"{video.get('width')}x{video.get('height')}",
    )

    fps_parts = str(video.get("avg_frame_rate", "0/1")).split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1] or 1)
    check("frame rate is 30", abs(fps - 30) < 0.5, f"{fps:.2f}")

    planned_ms = plan["plan"]["total_duration_ms"]
    actual_ms = float(fmt.get("duration", 0)) * 1000
    check(
        "duration matches the plan",
        abs(actual_ms - planned_ms) < 500,
        f"planned {planned_ms} ms, got {actual_ms:.0f} ms",
    )

    audio_ms = float(audio.get("duration", 0)) * 1000
    check(
        "audio is as long as the picture",
        abs(audio_ms - actual_ms) < 500,
        f"audio {audio_ms:.0f} ms vs video {actual_ms:.0f} ms",
    )
    check("audio is not empty", audio_ms > 1_000, f"{audio_ms:.0f} ms")

    # A full decode pass. Probing reads the header; this reads every packet,
    # which is the difference between "it opens" and "it plays".
    decode = subprocess.run(
        [ffmpeg(), "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        capture_output=True, text=True, timeout=600,
    )
    check("the file decodes completely", decode.returncode == 0, decode.stderr.strip()[:200])
    check("decoding reported no errors", decode.stderr.strip() == "", decode.stderr.strip()[:200])

    _report["output_duration_ms"] = round(actual_ms)
    _report["output_audio_ms"] = round(audio_ms)
    _report["output_dimensions"] = f"{video.get('width')}x{video.get('height')}"
    _report["output_fps"] = round(fps, 2)
    _report["ffmpeg"] = ffmpeg_version()


# ------------------------------------------------------------------------ main
def main() -> int:
    print("VisionForge - Phase 7 acceptance (music, beats and the audio mix)")
    print("=" * 60)

    try:
        health = httpx.get(f"{API}/health/ready", timeout=30).json()
    except httpx.HTTPError as exc:
        print(f"\nCannot reach the API at {API}: {exc}")
        return 2
    if health["status"] != "ok":
        print(f"\nAPI is not ready: {json.dumps(health, indent=2)}")
        return 2

    clips, track = build_inputs()
    project_id, media = scenario_prepare(clips, track)

    track_id = media[track.name]
    clip_id = media[clips[0].name]

    beats = scenario_analysis(project_id, track_id)
    plan = scenario_plan(project_id, track_id, beats)
    scenario_rejections(project_id, track_id, clip_id)
    render = scenario_render(project_id, plan)
    scenario_verify(project_id, render, plan)

    step("Summary")
    print(json.dumps(_report, indent=2, default=str))

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All Phase 7 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
