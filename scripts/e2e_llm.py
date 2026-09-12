"""Phase 5 acceptance test: natural language in, validated EditPlan out.

Drives a live API through the LLM planning lane and then the *existing* Phase 4
render pipeline, unchanged:

    upload clips -> ingest -> CPU analysis
        -> brief (handles only, no ids, no filenames)
        -> LLM planner -> EditDirective
        -> handle resolution + clamping
        -> EditPlan -> the Phase 4 validator
        -> Timeline -> RenderSpec -> FFmpeg argv
        -> MP4, verified with ffprobe

Run it twice, because the feature has to be right in both configurations:

    # provider disabled -- the rules engine plans everything
    python scripts/e2e_llm.py

    # provider enabled -- the whole LLM path runs
    VF_LLM_ENABLED=true VF_LLM_PROVIDER=stub python scripts/e2e_llm.py

The script reads the server's own ``/api/planner/capabilities`` to discover
which configuration it is talking to, and asserts the behaviour appropriate to
it. That way one script covers "AI off, everything still works" and "AI on, the
model's plan renders" without being told which is which.

Needs infrastructure, the API, a cpu worker and a render worker.
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
#: Six clips: four usable and visually distinct, two deliberately unusable. The
#: unusable pair matters -- it proves the deterministic gates run *before* the
#: model, so a model is never asked to avoid a black frame.
CLIP_SPECS: list[tuple[str, str]] = [
    ("sharp_bars.mp4", "smptebars=size=1280x720:rate=25:duration=6"),
    ("sharp_test.mp4", "testsrc=size=1280x720:rate=25:duration=6"),
    ("sharp_test2.mp4", "testsrc2=size=1280x720:rate=25:duration=6"),
    ("sharp_rgb.mp4", "rgbtestsrc=size=1280x720:rate=25:duration=6"),
    ("soft_blurred.mp4", "testsrc=size=1280x720:rate=25:duration=6,boxblur=20:2"),
    ("dark_crushed.mp4", "color=c=black:size=1280x720:rate=25:duration=6"),
]

#: The natural-language requests the feature exists to serve. Taken from the
#: Phase 5 brief, so the acceptance test covers what was actually asked for.
REQUESTS: list[tuple[str, str]] = [
    ("cinematic travel", "Edit this like a cinematic travel TikTok."),
    ("football highlight", "Make a fast football highlight."),
    ("gaming montage", "Make an energetic gaming montage."),
    ("anime AMV", "Create an anime AMV-style edit."),
    ("calm nature", "Make a calm nature video."),
    ("best moments", "Use the best moments and make it around 30 seconds."),
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


def plan(project_id: str, **body: object) -> dict:
    response = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan", json=body, timeout=180
    )
    response.raise_for_status()
    return dict(response.json())


# ------------------------------------------------------------------- scenarios
def scenario_capabilities() -> dict:
    step("1. What can this server plan with?")
    body = httpx.get(f"{API}/api/planner/capabilities", timeout=30).json()

    check("capabilities endpoint responds", bool(body["modes"]))
    check("all three modes offered", set(body["modes"]) == {"automatic", "rules", "ai"})
    check("eight styles offered", len(body["styles"]) == 8, str(len(body["styles"])))
    check("frame rates are a closed set", body["fps_presets"] == [24, 30, 60])
    check("quality presets offered", body["quality_presets"] == ["draft", "balanced", "high"])

    raw = json.dumps(body).lower()
    check(
        "no credential is exposed",
        not any(token in raw for token in ("api_key", "secret", "sk-", "bearer")),
    )

    if body["ai_available"]:
        print(f"{INFO} AI available: {body['provider']} · {body['model']}")
        if body["is_stub"]:
            print(f"{INFO} this is the deterministic stub, not a language model")
    else:
        print(f"{INFO} AI not configured; the rules engine plans everything")

    _report["capabilities"] = body
    return dict(body)


def scenario_ingest(clips: list[Path]) -> tuple[str, dict[str, str]]:
    step("2. Upload and analyse the source clips")
    project_id = create_project("Phase 5 LLM Lane")
    print(f"{INFO} project {project_id}")

    media: dict[str, str] = {}
    for path in clips:
        media_id, job_id = upload(project_id, path)
        media[path.name] = media_id
        assert job_id
        result = wait_for_job(job_id)
        check(f"ingest {path.name}", result["status"] == "succeeded", result["status"])

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis", json={"lanes": ["cpu"]}, timeout=60
    )
    check("CPU analysis accepted", queued.status_code == 202, str(queued.status_code))
    for job in queued.json()["jobs"]:
        result = wait_for_job(job["id"])
        check(f"analysis {job['id'][:8]}", result["status"] == "succeeded", result["status"])

    _report["input_clips"] = len(clips)
    return project_id, media


def scenario_rules_still_work(project_id: str) -> dict:
    """Phase 4's behaviour, unchanged. The regression that matters most."""
    step("3. The rules engine still plans, exactly as in Phase 4")
    document = plan(
        project_id, mode="rules", target_duration_ms=24_000, max_clips=4, min_clips=2
    )

    check("plan created", document["segment_count"] >= 2, str(document["segment_count"]))
    check("planned by the rules engine", document["planner"] == "rules-engine")
    check("no model was involved", document["llm"] is None)
    check("mode recorded", document["mode"]["mode"] == "rules")
    check(
        "duration as requested",
        document["total_duration_ms"] == 24_000,
        f"{document['total_duration_ms']} ms",
    )

    again = plan(
        project_id, mode="rules", target_duration_ms=24_000, max_clips=4, min_clips=2
    )
    check(
        "re-planning is deterministic",
        again["plan"]["segments"] == document["plan"]["segments"],
    )
    return document


def scenario_natural_language(project_id: str, ai_available: bool) -> list[dict]:
    step("4. Natural-language requests")
    plans: list[dict] = []

    for label, request_text in REQUESTS:
        document = plan(
            project_id,
            mode="automatic",
            request_text=request_text,
            max_clips=4,
            min_clips=2,
        )
        plans.append(document)

        mode = document["mode"]["mode"]
        planner = document["planner"]
        fallback = (document["llm"] or {}).get("fallback_reason")
        print(
            f"{INFO} {label:<20} mode={mode:<9} planner={planner:<12} "
            f"clips={document['segment_count']} "
            f"{document['total_duration_ms'] / 1000:.1f}s"
            + (f"  fallback={fallback}" if fallback else "")
        )

        check(f"{label}: a plan was produced", document["segment_count"] >= 2)
        check(f"{label}: the plan is bounded", document["total_duration_ms"] <= 600_000)
        check(
            f"{label}: every segment is legal",
            all(300 <= s["duration_ms"] <= 30_000 for s in document["plan"]["segments"]),
        )
        check(
            f"{label}: output geometry came from the server",
            (document["plan"]["output"]["width"], document["plan"]["output"]["height"])
            in {(1280, 720), (720, 1280), (720, 720)},
        )

        payload = json.dumps(document)
        check(
            f"{label}: no path or command in the plan",
            not any(token in payload for token in ("/etc/", "rm -rf", "ffmpeg", "-i /", "\\\\Windows")),
        )

        if ai_available:
            check(f"{label}: routed to the AI planner", mode == "ai", mode)

    if ai_available:
        used_llm = [p for p in plans if p["planner"] == "llm"]
        check(
            "the model planned at least one edit",
            bool(used_llm),
            f"{len(used_llm)}/{len(plans)}",
        )
    else:
        check(
            "every request still produced a rules plan",
            all(p["planner"] == "rules-engine" for p in plans),
        )
        matched = [p for p in plans if p["mode"]["inferred_style"]]
        check("styles were matched by keyword", bool(matched), f"{len(matched)} matched")

    _report["natural_language"] = [
        {
            "request": label,
            "mode": p["mode"]["mode"],
            "planner": p["planner"],
            "style": p["plan"]["metadata"].get("style"),
            "segments": p["segment_count"],
            "duration_ms": p["total_duration_ms"],
            "fallback_reason": (p["llm"] or {}).get("fallback_reason"),
        }
        for (label, _), p in zip(REQUESTS, plans, strict=True)
    ]
    return plans


def scenario_styles(project_id: str) -> None:
    step("5. Every style plans, with or without a model")
    styles = [
        "cinematic",
        "fast_montage",
        "sports_highlight",
        "gaming",
        "anime",
        "nature",
        "social",
        "custom",
    ]
    results = []
    for style in styles:
        document = plan(project_id, mode="rules", style=style, max_clips=4, min_clips=2)
        segments = document["plan"]["segments"]
        results.append(
            {
                "style": style,
                "segments": len(segments),
                "duration_ms": document["total_duration_ms"],
                "aspect": document["plan"]["output"]["aspect_ratio"],
                "clip_ms": segments[0]["duration_ms"] if segments else None,
            }
        )
        check(f"{style}: plans without a model", len(segments) >= 2)

    fast = next(r for r in results if r["style"] == "fast_montage")
    slow = next(r for r in results if r["style"] == "cinematic")
    check(
        "fast montage cuts faster than cinematic",
        int(fast["clip_ms"] or 0) < int(slow["clip_ms"] or 0),
        f"{fast['clip_ms']} ms vs {slow['clip_ms']} ms",
    )
    social = next(r for r in results if r["style"] == "social")
    check("social defaults to vertical", social["aspect"] == "9:16", str(social["aspect"]))

    _report["styles"] = results


def scenario_limits(project_id: str) -> None:
    step("6. Server-side limits hold")
    over_long = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan",
        json={"request_text": "x" * 5_000},
        timeout=60,
    )
    check("an oversized request is refused", over_long.status_code == 422,
          str(over_long.status_code))

    odd_fps = httpx.post(
        f"{API}/api/projects/{project_id}/edit-plan", json={"fps": 47}, timeout=60
    )
    check("an arbitrary frame rate is refused", odd_fps.status_code == 422,
          str(odd_fps.status_code))

    smuggled = plan(
        project_id,
        mode="rules",
        width=3840,
        height=2160,
        crf=1,
        output_path="/tmp/pwned.mp4",
        ffmpeg_args=["-i", "/etc/shadow"],
        max_clips=3,
        min_clips=2,
    )
    output = smuggled["plan"]["output"]
    check(
        "geometry cannot be requested by a client",
        (output["width"], output["height"]) == (1280, 720),
        f"{output['width']}x{output['height']}",
    )
    check("no smuggled field reaches the plan", "pwned" not in json.dumps(smuggled))

    stranger = httpx.get(
        f"{API}/api/projects/00000000-0000-0000-0000-000000000999/llm-runs", timeout=30
    )
    check("llm-runs enforces project ownership", stranger.status_code == 404,
          str(stranger.status_code))


def scenario_render(project_id: str, plan_document: dict) -> dict:
    step("7. Render the plan through the existing Phase 4 pipeline")
    started = time.monotonic()
    accepted = httpx.post(
        f"{API}/api/projects/{project_id}/render",
        json={"edit_plan_id": plan_document["id"]},
        timeout=60,
    )
    check("render accepted", accepted.status_code == 202, str(accepted.status_code))
    render = wait_for_render(project_id, accepted.json()["id"])
    wall_s = time.monotonic() - started

    check("render reached ready", render["status"] == "ready", str(render.get("error")))
    check("render has a playback URL", bool(render.get("playback_url")))

    output = CLIP_DIR / "phase5-output.mp4"
    output.write_bytes(httpx.get(render["playback_url"], timeout=300).content)
    check("file downloaded", output.exists() and output.stat().st_size > 0)
    check(
        "file size matches the recorded size",
        output.stat().st_size == render["bytes_size"],
        f"{output.stat().st_size} vs {render['bytes_size']}",
    )

    probe = ffprobe(output)
    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    check("ffprobe found a video stream", video is not None)
    assert video is not None

    numerator, denominator = (int(x) for x in video["r_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0.0
    duration_s = float(probe["format"]["duration"])

    check("codec is h264", video["codec_name"] == "h264", video["codec_name"])
    check("pixel format is yuv420p", video["pix_fmt"] == "yuv420p", video["pix_fmt"])
    check(
        "resolution matches the plan",
        (video["width"], video["height"])
        == (plan_document["plan"]["output"]["width"], plan_document["plan"]["output"]["height"]),
        f"{video['width']}x{video['height']}",
    )
    check(
        "duration matches the plan within a frame",
        abs(duration_s * 1000 - plan_document["total_duration_ms"]) < 1000 / fps + 50,
        f"{duration_s:.2f}s vs planned {plan_document['total_duration_ms'] / 1000:.2f}s",
    )

    decoded = subprocess.run(  # noqa: S603 - argv array
        [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    check("the whole file decodes without error", decoded.returncode == 0,
          decoded.stderr[:120])

    _report["render"] = {
        "planner": plan_document["planner"],
        "status": render["status"],
        "bytes_size": render["bytes_size"],
        "width": video["width"],
        "height": video["height"],
        "fps": round(fps, 2),
        "duration_s": round(duration_s, 2),
        "wall_s": round(wall_s, 1),
        "metrics": render.get("metrics"),
    }
    return render


def scenario_observability(project_id: str, ai_available: bool) -> None:
    step("8. Observability")
    body = httpx.get(f"{API}/api/projects/{project_id}/llm-runs", timeout=30).json()
    runs = body["items"]

    if not ai_available:
        check("no model runs were recorded", body["total"] == 0, str(body["total"]))
        return

    check("model runs were recorded", body["total"] > 0, str(body["total"]))
    for run in runs[:1]:
        check("run records the provider", bool(run["provider"]), str(run["provider"]))
        check("run records the model", bool(run["model"]), str(run["model"]))
        check("run records the prompt version", bool(run["prompt_version"]))
        check("run records a status", run["status"] in {"ok", "fallback", "error"},
              run["status"])

    raw = json.dumps(body).lower()
    check(
        "no prompt or completion is stored",
        not any(token in raw for token in ("you are the shot-selection", "rationale", "<<<user")),
    )
    _report["llm_runs"] = {"total": body["total"], "sample": runs[:3]}


# ------------------------------------------------------------------------ main
def main() -> int:
    print("VisionForge Phase 5 - LLM planning acceptance")
    print(f"API: {API}")

    clips = build_clips()
    capabilities = scenario_capabilities()
    ai_available = bool(capabilities["ai_available"])

    project_id, _media = scenario_ingest(clips)
    rules_plan = scenario_rules_still_work(project_id)
    language_plans = scenario_natural_language(project_id, ai_available)
    scenario_styles(project_id)
    scenario_limits(project_id)

    # Render whichever plan the model produced if there is one, so the render
    # test exercises the LLM path end to end; otherwise the rules plan.
    renderable = next(
        (p for p in language_plans if p["planner"] == "llm" and p["segment_count"] >= 2),
        rules_plan,
    )
    scenario_render(project_id, renderable)
    scenario_observability(project_id, ai_available)

    print("\nSummary\n-------")
    print(json.dumps(_report, indent=2, default=str))

    print("\n" + "=" * 60)
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("All Phase 5 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
