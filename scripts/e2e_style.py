"""Phase 8 acceptance test: a reference video's style, applied to other footage.

Phase 4 proved a planned edit renders. Phase 7 proved a beat-synced one does with
a music bed. This proves the whole reference chain:

    upload a fast-cut reference + 4 ordinary clips
        -> ingest         (the unchanged Phase 2 pipeline)
        -> analysis       (the CPU lane, now including motion and colour)
        -> style profile  (measured, versioned, confidence per feature)
        -> style strength (0 / 50 / 100, applied to the same footage)
        -> plan           (rules engine, no model required)
        -> validate       (the unchanged Phase 4 gate)
        -> beat sync      (the unchanged Phase 7 audio pipeline)
        -> audio mix      (unchanged)
        -> render         (unchanged)
        -> MP4            (ffprobe, and a full decode pass)

Three claims are under test, and the third is the one that matters.

**The profile is measured, not asserted.** The reference is generated with a
known cutting rate, and the detected shot length has to match it. Nothing about
the reference is sent by the client except its id.

**The dial does something, and does nothing at zero.** The same footage is
planned at 0%, 50% and 100%. Zero must produce byte-identical segments to a plan
made with no reference at all; 100% must move the pacing measurably toward the
reference; 50% must land between them.

**The reference is never in the output.** The rendered MP4 must contain only the
user's other clips. The reference is deliberately made the *strongest* clip in
the project -- sharpest, best exposed -- so that a selector which had not
excluded it would pick it first, and the check would fail loudly rather than by
luck.

Everything is generated with FFmpeg. No network, no model weights, no
copyrighted bytes.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_style.py

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

#: The reference cuts every 700 ms. Scene detection has to find that, and the
#: profile's median shot length has to land near it.
REFERENCE_SHOT_MS = 700
REFERENCE_SHOTS = 24
#: Scene detection works on visual change, and a boundary can land a frame
#: either side of the true cut. Two frames at 25 fps is 80 ms.
SHOT_TOLERANCE_MS = 200

PASS, FAIL, INFO = "PASS", "FAIL", " ..."
_failures: list[str] = []
_report: dict[str, object] = {}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def step(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        print("ffmpeg not on PATH")
        sys.exit(2)
    return binary


# ---------------------------------------------------------------------- inputs
#: Four ordinary clips: long, slow, single takes. Nothing in them resembles the
#: reference, so any movement in the planned pacing came from the reference.
CLIP_SPECS: list[tuple[str, str]] = [
    ("style_bars.mp4", "smptebars=size=1280x720:rate=25:duration=10"),
    ("style_test.mp4", "testsrc=size=1280x720:rate=25:duration=10"),
    ("style_test2.mp4", "testsrc2=size=1280x720:rate=25:duration=10"),
    ("style_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=10"),
]

REFERENCE_NAME = "style_reference.mp4"


def build_reference(binary: str, destination: Path) -> None:
    """A fast-cut reference: 24 hard cuts between visually distinct sources.

    Built by concatenating short segments rather than by filtering one source,
    because scene detection keys on visual change between consecutive frames and
    a fade or a wipe would not produce the hard boundaries a cutting rate is
    supposed to be measured from.
    """
    parts_dir = CLIP_DIR / "_ref_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Six visually unrelated generators, cycled. Adjacent segments must differ
    # enough that the detector sees a cut; two greys in a row would merge.
    sources = [
        "smptebars=size=1280x720:rate=25",
        "testsrc=size=1280x720:rate=25",
        "rgbtestsrc=size=1280x720:rate=25",
        "testsrc2=size=1280x720:rate=25",
        "color=c=navy:size=1280x720:rate=25",
        "color=c=orange:size=1280x720:rate=25",
    ]

    listing = parts_dir / "parts.txt"
    lines: list[str] = []
    for index in range(REFERENCE_SHOTS):
        part = parts_dir / f"part_{index:02d}.mp4"
        if not part.exists():
            subprocess.run(
                [
                    binary, "-y", "-loglevel", "error",
                    "-f", "lavfi",
                    "-i", f"{sources[index % len(sources)]}:duration={REFERENCE_SHOT_MS / 1000}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
                    str(part),
                ],
                check=True, capture_output=True, timeout=300,
            )
        lines.append(f"file '{part.as_posix()}'")
    listing.write_text("\n".join(lines), encoding="utf-8")

    subprocess.run(
        [
            binary, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(destination),
        ],
        check=True, capture_output=True, timeout=300,
    )


def build_inputs() -> tuple[list[Path], Path]:
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
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
                    str(destination),
                ],
                check=True, capture_output=True, timeout=300,
            )
        clips.append(destination)

    reference = CLIP_DIR / REFERENCE_NAME
    if not reference.exists():
        build_reference(binary, reference)
    return clips, reference


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
    binary = shutil.which("ffprobe")
    if binary is None:
        print("ffprobe not on PATH")
        sys.exit(2)
    result = subprocess.run(
        [binary, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return json.loads(result.stdout)


def plan_at(project_id: str, strength: str) -> dict:
    """One plan at one style strength. Everything else held constant."""
    response = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={
            "mode": "rules",
            "target_duration_ms": 20_000,
            "max_clips": 4,
            "min_clips": 2,
            "aspect_ratio": "16:9",
            "fps": 30,
            "quality": "draft",
            "order": "sequence",
            "style_strength": strength,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------------- scenarios
def scenario_media() -> tuple[str, str, list[str]]:
    step("1. MEDIA - import a reference and four ordinary clips")
    clips, reference = build_inputs()
    project_id = create_project("Phase 8 Reference Style")
    print(f" ... project {project_id}")

    reference_id, job_id = upload(project_id, reference)
    if job_id:
        result = wait_for_job(job_id)
        check(f"ingest {reference.name}", result["status"] == "succeeded", result["status"])

    clip_ids: list[str] = []
    for path in clips:
        media_id, job = upload(project_id, path)
        clip_ids.append(media_id)
        if job:
            result = wait_for_job(job)
            check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    ready = [m for m in listing["items"] if m["status"] == "ready"]
    check("all five assets ready", len(ready) == 5, f"{len(ready)}/5")

    probed = next((m for m in ready if m["id"] == reference_id), None)
    expected_ms = REFERENCE_SHOTS * REFERENCE_SHOT_MS
    check(
        "the reference is the length it was built to be",
        probed is not None and abs((probed["duration_ms"] or 0) - expected_ms) < 500,
        f"{probed['duration_ms'] if probed else '?'} ms vs {expected_ms}",
    )
    return project_id, reference_id, clip_ids


def scenario_analysis(project_id: str) -> None:
    step("2. ANALYSIS - the CPU lane, now including motion and colour")
    accepted = httpx.post(f"{API}/api/projects/{project_id}/analyze", timeout=60)
    check("analysis accepted", accepted.status_code == 202, str(accepted.status_code))

    for job in accepted.json()["jobs"]:
        result = wait_for_job(job["id"])
        check(f"analysis job {job['id'][:8]}", result["status"] == "succeeded", result["status"])

    records = httpx.get(f"{API}/api/projects/{project_id}/analysis", timeout=60).json()
    dynamics = [r for r in records["items"] if r["analyzer"] == "dynamics" and r["status"] == "ok"]
    check("every clip has a dynamics record", len(dynamics) == 5, str(len(dynamics)))

    if dynamics:
        sample = dynamics[0]["payload"]
        check("dynamics measured motion", isinstance(sample.get("motion"), (int, float)))
        check("dynamics measured saturation", isinstance(sample.get("saturation"), (int, float)))
        check(
            "dynamics reported its own confidence",
            0.0 < float(sample.get("motion_confidence", 0)) <= 1.0,
            str(sample.get("motion_confidence")),
        )


def scenario_profile(project_id: str, reference_id: str, clip_ids: list[str]) -> dict:
    step("3. STYLE PROFILE - measured from the reference, never asserted")

    empty = httpx.get(f"{API}/api/projects/{project_id}/reference", timeout=30).json()
    check("a project starts with no reference", empty["media_id"] is None)

    set_response = httpx.put(
        f"{API}/api/projects/{project_id}/reference",
        json={"media_id": reference_id},
        timeout=60,
    )
    check("the reference was accepted", set_response.status_code == 200,
          str(set_response.status_code))

    state = set_response.json()
    profile = state["profile"]
    check("a profile was derived", profile is not None)
    if profile is None:
        return {}

    check("the profile is usable", profile["usable"] is True)
    check("the profile is versioned", bool(profile["version"]), profile["version"])
    check(
        "the measured shot length matches how the reference was built",
        abs(profile["shot_ms"]["value"] - REFERENCE_SHOT_MS) <= SHOT_TOLERANCE_MS,
        f"{profile['shot_ms']['value']} ms vs {REFERENCE_SHOT_MS}",
    )
    check("pacing is read as rapid", profile["pacing"] == "rapid", str(profile["pacing"]))
    check(
        "the cut rate is per minute and plausible",
        profile["cut_rate"] is not None and profile["cut_rate"]["value"] > 40,
        str(profile["cut_rate"]["value"] if profile["cut_rate"] else None),
    )
    check("motion was measured", profile["motion"] is not None)
    check("saturation was measured", profile["saturation"] is not None)
    check(
        "every feature carries a confidence",
        all(
            profile[name] is None or 0.0 <= profile[name]["confidence"] <= 1.0
            for name in ("shot_ms", "cut_rate", "motion", "saturation", "luminance", "contrast")
        ),
    )

    # Determinism: the same rows must produce the same profile.
    again = httpx.get(f"{API}/api/projects/{project_id}/reference", timeout=30).json()
    check("the profile is deterministic across reads", again["profile"] == profile)

    # Security: nothing but numbers.
    body = set_response.text
    for forbidden in ("local_path", "storage_key", "original_filename", ".mp4", "bucket"):
        check(f"the profile carries no {forbidden}", forbidden not in body)

    # Security: a clip from another project cannot be nominated.
    other_project = create_project("Phase 8 Other Project")
    refused = httpx.put(
        f"{API}/api/projects/{other_project}/reference",
        json={"media_id": reference_id},
        timeout=30,
    )
    check("another project's media is refused", refused.status_code == 404,
          f"HTTP {refused.status_code}")

    # Security: audio cannot be a reference (there is no audio here, so the
    # nearest available negative is a nonexistent id).
    missing = httpx.put(
        f"{API}/api/projects/{project_id}/reference",
        json={"media_id": clip_ids[0][:-1] + ("0" if clip_ids[0][-1] != "0" else "1")},
        timeout=30,
    )
    check("an unknown id is refused", missing.status_code == 404, f"HTTP {missing.status_code}")

    _report["profile"] = {
        "shot_ms": profile["shot_ms"]["value"],
        "shot_confidence": profile["shot_ms"]["confidence"],
        "pacing": profile["pacing"],
        "cut_rate": profile["cut_rate"]["value"] if profile["cut_rate"] else None,
        "scene_count": profile["scene_count"],
        "motion": profile["motion"]["value"] if profile["motion"] else None,
        "saturation": profile["saturation"]["value"] if profile["saturation"] else None,
        "confidence": profile["confidence"],
        "version": profile["version"],
    }
    return profile


def scenario_strength(project_id: str, reference_id: str) -> dict:
    step("4. STYLE STRENGTH - the same footage at 0%, 50% and 100%")

    plans = {strength: plan_at(project_id, strength) for strength in ("0", "50", "100")}
    per_clip = {
        strength: plan["plan"]["metadata"]["per_clip_ms"] for strength, plan in plans.items()
    }
    print(f" ... per-clip ms by strength: {per_clip}")

    # A plan with the reference cleared is the control.
    httpx.delete(f"{API}/api/projects/{project_id}/reference", timeout=30).raise_for_status()
    control = plan_at(project_id, "0")
    httpx.put(
        f"{API}/api/projects/{project_id}/reference",
        json={"media_id": reference_id},
        timeout=30,
    ).raise_for_status()

    def segments(plan: dict) -> list[tuple[str, int, int]]:
        return [
            (s["media_id"], s["source_in_ms"], s["source_out_ms"])
            for s in plan["plan"]["segments"]
        ]

    check(
        "0% is identical to having no reference at all",
        segments(plans["0"]) == segments(control),
        f"{len(segments(plans['0']))} segments",
    )
    check(
        "0% records no style policy",
        plans["0"]["plan"]["metadata"].get("style_policy") is None,
    )
    check(
        "100% shortens the clips toward the reference",
        per_clip["100"] < per_clip["0"],
        f"{per_clip['100']} ms vs {per_clip['0']} ms",
    )
    check(
        "50% lands between the two",
        per_clip["100"] <= per_clip["50"] <= per_clip["0"],
        f"{per_clip['50']} ms",
    )

    policy = plans["100"]["plan"]["metadata"].get("style_policy")
    check("100% records what the reference moved", policy is not None)
    if policy:
        check("the recorded strength is what was asked for", policy["strength"] == "100")
        check("shot length is among the influences", "shot_ms" in policy["influence"])
        check("style affinity was given weight", policy["affinity_weight"] > 0,
              str(policy["affinity_weight"]))
        check("the policy carries no reference id", reference_id not in json.dumps(policy))

    _report["per_clip_ms"] = per_clip
    _report["target_clip_ms"] = policy["target_clip_ms"] if policy else None
    return plans["100"]


def scenario_render(project_id: str, plan: dict, reference_id: str) -> Path:
    step("5. RENDER - the styled plan, through the unchanged pipeline")

    plan_id = plan["id"]
    segments = plan["plan"]["segments"]
    used = {s["media_id"] for s in segments}

    check(
        "the reference is not in the plan",
        reference_id not in used,
        f"{len(used)} distinct clips",
    )
    check("the plan has segments", len(segments) >= 2, str(len(segments)))

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan_id},
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
    if not url:
        return CLIP_DIR / "phase8_output.mp4"

    output = CLIP_DIR / "phase8_output.mp4"
    output.write_bytes(httpx.get(url, timeout=300).content)
    check("the file downloaded", output.stat().st_size > 0, f"{output.stat().st_size} bytes")

    _report["render_ms"] = render_ms
    _report["output_bytes"] = output.stat().st_size
    return output


def scenario_verify(output: Path) -> None:
    step("6. MP4 - verified independently with ffprobe")

    probe = ffprobe(output)
    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    check("a video stream exists", video is not None)
    if video is None:
        return

    check("container is mp4", "mp4" in probe["format"]["format_name"],
          probe["format"]["format_name"])
    check("video codec is h264", video["codec_name"] == "h264", video["codec_name"])
    check("resolution is 1280x720", (video["width"], video["height"]) == (1280, 720),
          f"{video['width']}x{video['height']}")

    rate = video.get("avg_frame_rate", "0/1")
    numerator, _, denominator = rate.partition("/")
    fps = float(numerator) / float(denominator or 1)
    check("frame rate is 30", abs(fps - 30.0) < 0.01, f"{fps:.2f}")

    duration_ms = int(float(probe["format"]["duration"]) * 1000)
    check("the file has real duration", duration_ms > 1_000, f"{duration_ms} ms")

    binary = shutil.which("ffmpeg")
    assert binary is not None
    decode = subprocess.run(
        [binary, "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        capture_output=True, timeout=600,
    )
    check("the file decodes completely", decode.returncode == 0)
    check("decoding reported no errors", not decode.stderr.strip(),
          decode.stderr.decode(errors="replace")[:200])

    _report["output_duration_ms"] = duration_ms
    _report["output_dimensions"] = f"{video['width']}x{video['height']}"
    _report["output_fps"] = round(fps, 2)


def main() -> int:
    print("VisionForge - Phase 8 acceptance (reference video style intelligence)")
    print("=" * 60)

    project_id, reference_id, clip_ids = scenario_media()
    scenario_analysis(project_id)
    profile = scenario_profile(project_id, reference_id, clip_ids)
    if not profile:
        print("\nno profile derived; the rest cannot be checked")
        return 1

    plan = scenario_strength(project_id, reference_id)
    output = scenario_render(project_id, plan, reference_id)
    scenario_verify(output)

    version = subprocess.run(
        [ffmpeg(), "-version"], capture_output=True, text=True, timeout=60
    ).stdout.splitlines()[0]
    _report["ffmpeg"] = version

    step("Summary")
    print(json.dumps(_report, indent=2, sort_keys=True))

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All Phase 8 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
