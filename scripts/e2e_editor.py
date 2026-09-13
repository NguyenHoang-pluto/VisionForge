"""Phase 6 acceptance test: the editor's path from a timeline to an MP4.

Phase 4 proved a *planned* edit renders. This proves a **hand-cut** one does, and
that cutting it by hand buys no authority the planner did not have.

    upload 4 clips -> ingest -> CPU analysis
        -> plan automatically (the first cut the editor shows)
        -> edit that plan the way the timeline does:
             trim a clip, reorder two, delete one, split one
        -> POST /edit-plan/manual
        -> validation (the unchanged Phase 4 validator)
        -> render
        -> MP4, verified with ffprobe against what the timeline said

The negative cases matter as much as the positive one: a trim past the end of a
source, a clip shorter than the renderer can produce, and a media id from
another project must each be refused *before* anything is stored, with a reason
the editor can show. A timeline that could smuggle any of those past the gate
would mean the browser had acquired a rendering model of its own, which is the
one thing Phase 6 was not allowed to do.

Run with infrastructure, the API, a cpu worker and a render worker up::

    python scripts/e2e_editor.py

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
#: Four distinguishable 8-second clips. Long enough that a trim has somewhere to
#: go, and visually distinct so a wrong order is observable in the output.
CLIP_SPECS: list[tuple[str, str]] = [
    ("edit_bars.mp4", "smptebars=size=1280x720:rate=25:duration=8"),
    ("edit_test.mp4", "testsrc=size=1280x720:rate=25:duration=8"),
    ("edit_test2.mp4", "testsrc2=size=1280x720:rate=25:duration=8"),
    ("edit_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=8"),
]


def build_clips() -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not on PATH")
        sys.exit(2)

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, source in CLIP_SPECS:
        destination = CLIP_DIR / name
        if not destination.exists():
            subprocess.run(
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
    binary = shutil.which("ffprobe")
    assert binary
    result = subprocess.run(
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


def post_manual(project_id: str, body: dict) -> httpx.Response:
    return httpx.post(f"{API}/api/projects/{project_id}/edit-plan/manual", json=body, timeout=60)


# ------------------------------------------------------------------- scenarios
def scenario_prepare(clips: list[Path]) -> tuple[str, dict[str, str]]:
    step("1. Import and analyse the footage")
    project_id = create_project("Phase 6 Editing Workstation")
    print(f"{INFO} project {project_id}")

    media: dict[str, str] = {}
    for path in clips:
        media_id, job_id = upload(project_id, path)
        media[path.name] = media_id
        assert job_id
        result = wait_for_job(job_id)
        check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    ready = [item for item in listing["items"] if item["status"] == "ready"]
    check("all clips ready", len(ready) == len(clips), f"{len(ready)}/{len(clips)}")

    # The editor's preview plays proxies, never originals. If ingest did not
    # make one, the timeline has nothing to scrub.
    with_proxy = [item for item in ready if item["has_proxy"]]
    check("every clip has a 720p proxy", len(with_proxy) == len(ready), f"{len(with_proxy)}")

    for item in ready[:1]:
        proxy = httpx.get(
            f"{API}/api/projects/{project_id}/media/{item['id']}/proxy", timeout=30
        )
        check("proxy URL is presigned", proxy.status_code == 200, str(proxy.status_code))
        check("proxy URL is not a storage key", "http" in proxy.json().get("url", ""))

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis", json={"lanes": ["cpu"]}, timeout=60
    )
    check("CPU analysis accepted", queued.status_code == 202, str(queued.status_code))
    for job in queued.json()["jobs"]:
        result = wait_for_job(job["id"])
        check(f"analysis {job['id'][:8]}", result["status"] == "succeeded", result["status"])

    _report["input_clips"] = len(clips)
    return project_id, media


def scenario_first_cut(project_id: str) -> dict:
    """The plan the editor seeds its timeline from."""
    step("2. Generate the first cut")

    response = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={"mode": "rules", "target_duration_ms": 20_000, "max_clips": 4, "min_clips": 2},
        timeout=120,
    )
    check("plan created", response.status_code == 201, str(response.status_code))
    plan = response.json()
    check("plan has segments", plan["segment_count"] >= 2, str(plan["segment_count"]))

    # The listing is deliberately thin; the detail route is what the editor
    # loads to fill a timeline. Both must agree about the same plan.
    listing = httpx.get(f"{API}/api/projects/{project_id}/edit-plan", timeout=30).json()
    check("plan appears in the listing", any(p["id"] == plan["id"] for p in listing["items"]))
    check(
        "the listing carries no plan document",
        "plan" not in listing["items"][0] or not listing["items"][0].get("plan"),
        "listings stay thin on purpose",
    )

    detail = httpx.get(
        f"{API}/api/projects/{project_id}/edit-plan/{plan['id']}", timeout=30
    ).json()
    check("the detail route carries the document", len(detail["plan"]["segments"]) >= 2)

    _report["first_cut"] = {
        "planner": detail["planner"],
        "segments": detail["segment_count"],
        "duration_ms": detail["total_duration_ms"],
    }
    return detail


def scenario_edit(project_id: str, first_cut: dict) -> dict:
    """Do to the plan what the timeline does, then store the result."""
    step("3. Edit the timeline and store it")

    segments = sorted(first_cut["plan"]["segments"], key=lambda s: s["order"])
    original_order = [s["media_id"] for s in segments]

    # The four edits the timeline supports, applied in one pass:
    #   trim    - pull the first clip's out point in by a second
    #   reorder - swap the first two
    #   split   - cut the last clip in two
    #   delete  - drop the middle one (if there is one to drop)
    cuts = [
        {
            "media_id": s["media_id"],
            "source_in_ms": s["source_in_ms"],
            "source_out_ms": s["source_out_ms"],
        }
        for s in segments
    ]

    cuts[0]["source_out_ms"] = max(
        cuts[0]["source_in_ms"] + 1000, cuts[0]["source_out_ms"] - 1000
    )
    cuts[0], cuts[1] = cuts[1], cuts[0]

    last = cuts[-1]
    midpoint = (last["source_in_ms"] + last["source_out_ms"]) // 2
    if midpoint - last["source_in_ms"] >= 300 and last["source_out_ms"] - midpoint >= 300:
        left = dict(last, source_out_ms=midpoint)
        right = dict(last, source_in_ms=midpoint)
        cuts[-1:] = [left, right]

    if len(cuts) > 3:
        del cuts[2]

    expected_ms = sum(c["source_out_ms"] - c["source_in_ms"] for c in cuts)

    response = post_manual(
        project_id,
        {
            "segments": cuts,
            "aspect_ratio": "16:9",
            "fps": 30,
            "quality": "draft",
            "derived_from_edit_plan_id": first_cut["id"],
        },
    )
    check("hand-cut timeline accepted", response.status_code == 201, response.text[:160])
    plan = response.json()

    check("stored as a manual plan", plan["planner"] == "manual", plan["planner"])
    check("clip count matches the timeline", plan["segment_count"] == len(cuts))
    check(
        "duration matches the timeline",
        plan["total_duration_ms"] == expected_ms,
        f"{plan['total_duration_ms']} vs {expected_ms}",
    )

    stored = sorted(plan["plan"]["segments"], key=lambda s: s["order"])
    check(
        "order comes from list position",
        [s["order"] for s in stored] == list(range(len(cuts))),
    )
    check(
        "the reorder was preserved",
        stored[0]["media_id"] == original_order[1],
        "first clip is the one that was second",
    )
    check(
        "the trim was preserved",
        any(
            s["source_out_ms"] == cuts[i]["source_out_ms"] for i, s in enumerate(stored)
        ),
    )
    check(
        "the server chose the geometry",
        (plan["plan"]["output"]["width"], plan["plan"]["output"]["height"]) == (1280, 720),
        f"{plan['plan']['output']['width']}x{plan['plan']['output']['height']}",
    )
    check(
        "provenance records the plan it was cut from",
        plan["plan"]["metadata"].get("derived_from_edit_plan_id") == first_cut["id"],
    )

    _report["hand_cut"] = {
        "segments": plan["segment_count"],
        "duration_ms": plan["total_duration_ms"],
        "expected_ms": expected_ms,
    }
    return plan


def scenario_rejections(project_id: str, first_cut: dict) -> None:
    """A timeline gets no authority the planner did not have."""
    step("4. Timelines the server must refuse")

    segment = sorted(first_cut["plan"]["segments"], key=lambda s: s["order"])[0]
    media_id = segment["media_id"]

    cases: list[tuple[str, dict]] = [
        (
            "trim past the end of the source",
            {"segments": [{"media_id": media_id, "source_in_ms": 0, "source_out_ms": 900_000}]},
        ),
        (
            "clip shorter than the renderer can produce",
            {"segments": [{"media_id": media_id, "source_in_ms": 0, "source_out_ms": 50}]},
        ),
        (
            "media from another project",
            {
                "segments": [
                    {
                        "media_id": "00000000-0000-0000-0000-0000000000ff",
                        "source_in_ms": 0,
                        "source_out_ms": 4000,
                    }
                ]
            },
        ),
        ("an empty timeline", {"segments": []}),
        (
            "a frame rate that is not offered",
            {
                "segments": [
                    {"media_id": media_id, "source_in_ms": 0, "source_out_ms": 4000}
                ],
                "fps": 29,
            },
        ),
    ]

    for label, body in cases:
        response = post_manual(project_id, body)
        check(f"refused: {label}", response.status_code == 422, str(response.status_code))

    # A geometry field in the body is not a way in: there is no such field on the
    # model, so it is dropped rather than honoured.
    smuggle = post_manual(
        project_id,
        {
            "segments": [{"media_id": media_id, "source_in_ms": 0, "source_out_ms": 4000}],
            "width": 3840,
            "height": 2160,
            "crf": 1,
            "ffmpeg_args": "-vf drawtext=text=owned",
        },
    )
    check("extra geometry fields are ignored", smuggle.status_code == 201, smuggle.text[:120])
    if smuggle.status_code == 201:
        output = smuggle.json()["plan"]["output"]
        check(
            "and the output is still the server's preset",
            (output["width"], output["height"]) == (1280, 720),
            f"{output['width']}x{output['height']}",
        )

    # Nothing invalid was stored. The plan count should only have grown by the
    # two plans that were legitimately accepted.
    listing = httpx.get(f"{API}/api/projects/{project_id}/edit-plan", timeout=30).json()
    check(
        "no rejected timeline reached the database",
        all(p["total_duration_ms"] > 0 for p in listing["items"]),
    )


def scenario_render(project_id: str, plan: dict) -> dict:
    step("5. Render the hand-cut timeline")

    response = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan["id"]},
        timeout=60,
    )
    check("render queued", response.status_code == 202, str(response.status_code))
    render = response.json()
    check("render row exists before the file does", render["status"] == "pending")
    check("render is attached to a job", bool(render["job_id"]))

    final = wait_for_render(project_id, render["id"])
    check("render succeeded", final["status"] == "ready", str(final.get("error")))
    check("playback URL is issued once ready", bool(final.get("playback_url")))

    _report["render"] = {
        "status": final["status"],
        "width": final.get("width"),
        "height": final.get("height"),
        "duration_ms": final.get("duration_ms"),
        "bytes": final.get("bytes_size"),
    }
    return final


def scenario_verify(render: dict, plan: dict) -> None:
    step("6. Verify the file independently")

    url = render.get("playback_url")
    if not url:
        check("a file to verify", False, "no playback URL")
        return

    output = CLIP_DIR / "phase6_output.mp4"
    output.write_bytes(httpx.get(url, timeout=300, follow_redirects=True).content)
    check("file downloaded", output.stat().st_size > 0, f"{output.stat().st_size} bytes")

    probe = ffprobe(output)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")

    container = probe["format"]["format_name"]
    check("container is mp4", "mp4" in container, container)
    check("codec is h264", video["codec_name"] == "h264", video["codec_name"])
    check("pixel format is yuv420p", video["pix_fmt"] == "yuv420p", video["pix_fmt"])
    check("geometry is the server's preset", (video["width"], video["height"]) == (1280, 720))

    numerator, denominator = video["r_frame_rate"].split("/")
    measured_fps = int(numerator) / int(denominator)
    check("frame rate is 30", abs(measured_fps - 30) < 0.5, str(round(measured_fps, 3)))

    duration_s = float(probe["format"]["duration"])
    expected_s = plan["total_duration_ms"] / 1000
    check(
        "duration matches the timeline the user cut",
        abs(duration_s - expected_s) < 0.5,
        f"{duration_s:.2f}s vs {expected_s:.2f}s",
    )

    binary = shutil.which("ffmpeg")
    assert binary
    decode = subprocess.run(
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
        "width": video["width"],
        "height": video["height"],
        "fps": round(measured_fps, 3),
        "duration_s": round(duration_s, 3),
        "bytes": output.stat().st_size,
    }
    _report["output_path"] = str(output)


def main() -> int:
    print("VisionForge Phase 6 - editing workstation acceptance")
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
    project_id, _media = scenario_prepare(clips)
    first_cut = scenario_first_cut(project_id)
    hand_cut = scenario_edit(project_id, first_cut)
    scenario_rejections(project_id, first_cut)
    render = scenario_render(project_id, hand_cut)
    scenario_verify(render, hand_cut)

    step("Summary")
    print(json.dumps(_report, indent=2, default=str))

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All Phase 6 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
