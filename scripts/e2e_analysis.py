"""Phase 3 analysis acceptance test.

Drives a live API and live CPU + GPU Celery workers through the deterministic
analysis flow Phase 3 was asked to prove:

    upload image / video / audio
        -> ingest (proxy, thumbnail)
        -> CPU analysis  (quality, scenes, pHash)
        -> GPU analysis  (CLIP embedding, face detection)
        -> pgvector similarity

Expectations per medium:

    image  quality, blur, exposure, contrast, pHash, CLIP, faces
    video  metadata, scenes, representative-frame quality, pHash, CLIP, faces
    audio  recognised as unsupported -- never a failure

Run with infrastructure, the API and both workers already up::

    python scripts/e2e_analysis.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

API = os.environ.get("VF_API_URL", "http://127.0.0.1:8000")
FIXTURES = Path(__file__).resolve().parents[1] / "apps/api/tests/integration/fixtures"
JOB_TIMEOUT_S = 600

PASS, FAIL, INFO = "PASS", "FAIL", " ..."
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{PASS if condition else FAIL}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def step(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


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

    httpx.put(body["upload_url"], content=path.read_bytes(), timeout=180).raise_for_status()

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


def analysis_for(project_id: str, media_id: str) -> dict[str, dict]:
    response = httpx.get(
        f"{API}/api/projects/{project_id}/media/{media_id}/analysis", timeout=30
    )
    response.raise_for_status()
    return {row["analyzer"]: row for row in response.json()["items"]}


# ------------------------------------------------------------------- scenarios
def scenario_pipeline() -> tuple[str, dict[str, str]]:
    step("1. Ingest and analyse image, video and audio")
    project_id = create_project("Phase 3 Analysis Acceptance")
    print(f"{INFO} project {project_id}")

    media: dict[str, str] = {}
    for name in ("image.jpg", "video_1080.mp4", "audio.mp3"):
        media_id, job_id = upload(project_id, FIXTURES / name)
        media[name] = media_id
        assert job_id
        result = wait_for_job(job_id)
        check(f"ingest {name}", result["status"] == "succeeded", result["status"])

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis", json={}, timeout=60
    )
    check("analysis accepted (202)", queued.status_code == 202, str(queued.status_code))
    body = queued.json()
    check(
        f"queued {body['queued']} jobs for {body['media_count']} assets",
        body["media_count"] == 3 and body["queued"] == 6,
        f"lanes={body['lanes']}",
    )

    for job in body["jobs"]:
        result = wait_for_job(job["id"])
        check(
            f"analysis job {job['id'][:8]} ({job['type'].replace('media_analyze_', '')})",
            result["status"] == "succeeded",
            result["status"],
        )
    return project_id, media


#: Every analyzer that must record a verdict for a piece of media.
#:
#: By name rather than by count. A count passes when one analyzer disappears and
#: another arrives, which is the failure it exists to catch. Phase 3 shipped
#: five; ``beats`` arrived in Phase 7 and ``dynamics`` in Phase 8, and both
#: record a verdict for every asset like the rest.
ANALYZERS = {"quality", "phash", "clip", "faces", "scenes", "beats", "dynamics"}

#: What an *audio* file gets. Everything that looks at pictures is unsupported,
#: which is a verdict and not a failure -- and ``beats`` is the exception,
#: because from Phase 7 detecting a tempo in audio is exactly its job.
AUDIO_OK = {"beats"}


def scenario_image(project_id: str, media_id: str) -> None:
    step("2. Image - quality, blur, exposure, contrast, pHash, CLIP, faces")
    rows = analysis_for(project_id, media_id)

    check("every analyzer recorded", set(rows) == ANALYZERS, ",".join(sorted(rows)))

    quality = rows["quality"]
    check("quality: ok", quality["status"] == "ok", quality["status"])
    p = quality["payload"]
    check("quality: blur score present", isinstance(p.get("blur_score"), (int, float)))
    check("quality: contrast present", isinstance(p.get("contrast"), (int, float)))
    check(
        "quality: exposure present",
        all(k in p for k in ("mean_luminance", "badly_exposed_frames")),
    )
    check("quality: thresholds recorded with the scores", "thresholds" in p)
    check("quality: frame-level detail kept", len(p.get("frames", [])) >= 1)

    phash = rows["phash"]
    check("phash: ok", phash["status"] == "ok")
    check("phash: 64-bit hex hash", len(phash["payload"]["phash"]) == 16)
    check("phash: aHash stored alongside", len(phash["payload"]["ahash"]) == 16)

    clip = rows["clip"]
    check("clip: ok", clip["status"] == "ok", clip["status"])
    check("clip: embedding stored", clip["payload"]["dim"] == 512 and clip["has_embedding"])
    check("clip: ran on the GPU", (clip["metrics"] or {}).get("device") == "cuda")
    check(
        "clip: VRAM measured",
        float((clip["metrics"] or {}).get("vram_peak_mb", 0)) > 0,
        f"{(clip['metrics'] or {}).get('vram_peak_mb')} MiB",
    )

    faces = rows["faces"]
    check("faces: ok", faces["status"] == "ok")
    check("faces: count reported", isinstance(faces["payload"]["face_count"], int))
    check("faces: no identity stored", faces["payload"]["identity_stored"] is False)

    check("image: scenes correctly unsupported", rows["scenes"]["status"] == "unsupported")


def scenario_video(project_id: str, media_id: str) -> None:
    step("3. Video - metadata, scenes, frame sampling, quality, CLIP, faces")
    rows = analysis_for(project_id, media_id)

    scenes = rows["scenes"]
    check("scenes: ok", scenes["status"] == "ok", scenes["status"])
    check("scenes: at least one boundary", scenes["payload"]["scene_count"] >= 1)
    first = (scenes["payload"].get("scenes") or [{}])[0]
    check(
        "scenes: structured boundaries",
        all(k in first for k in ("scene_id", "start_ms", "end_ms", "duration_ms")),
    )

    quality = rows["quality"]
    check("quality: ok", quality["status"] == "ok")
    check(
        "quality: several representative frames sampled",
        quality["payload"]["frame_count"] > 1,
        str(quality["payload"]["frame_count"]),
    )

    check("phash: ok", rows["phash"]["status"] == "ok")
    check("clip: ok", rows["clip"]["status"] == "ok")
    check("clip: embedding stored", rows["clip"]["has_embedding"] is True)
    check("faces: ok", rows["faces"]["status"] == "ok")

    # Proxy-first: a 1080p source gets a 720p proxy at ingest, and analysis
    # must have read that rather than the master.
    check(
        "proxy was analysed, not the original",
        quality["payload"].get("used_proxy") is True,
        str(quality["payload"].get("used_proxy")),
    )


def scenario_audio(project_id: str, media_id: str) -> None:
    step("4. Audio - unsupported, never a failure")
    rows = analysis_for(project_id, media_id)

    check("every analyzer recorded a verdict", set(rows) == ANALYZERS, ",".join(sorted(rows)))
    for name, row in sorted(rows.items()):
        if name in AUDIO_OK:
            # Beat detection on an audio file is the one analyzer that has
            # something to say here. "ok" is the pass; a failure is still a
            # failure.
            check(f"{name}: ok on audio", row["status"] == "ok", row["status"])
            continue
        check(
            f"{name}: unsupported (not failed)",
            row["status"] == "unsupported",
            row["status"],
        )
    check("audio: no embedding stored", rows["clip"]["has_embedding"] is False)


def scenario_similarity(project_id: str, media: dict[str, str]) -> None:
    step("5. pgvector similarity")
    duplicate_id, _ = upload(project_id, FIXTURES / "image.jpg")

    # The duplicate shares bytes with the first image, so ingest dedupes it.
    # Analyse whatever survived and search from the original.
    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis",
        json={"media_id": media["image.jpg"], "lanes": ["gpu"]},
        timeout=60,
    ).json()
    for job in queued["jobs"]:
        wait_for_job(job["id"])

    response = httpx.get(
        f"{API}/api/projects/{project_id}/media/{media['image.jpg']}/similar", timeout=60
    )
    check("similarity endpoint responds", response.status_code == 200, str(response.status_code))
    if response.status_code == 200:
        results = response.json()["results"]
        check("similarity returns ranked hits", isinstance(results, list))
        if results:
            check(
                "nearest hit has a real similarity score",
                0.0 <= results[0]["similarity"] <= 1.0,
                f"{results[0]['similarity']:.4f}",
            )
    del duplicate_id


def scenario_versioning(project_id: str, media_id: str) -> None:
    step("6. Analyzer versioning and idempotency")
    before = analysis_for(project_id, media_id)

    queued = httpx.post(
        f"{API}/api/projects/{project_id}/analysis",
        json={"media_id": media_id, "lanes": ["cpu"]},
        timeout=60,
    ).json()
    check(
        "re-requesting the same analysis is idempotent",
        queued["queued"] == 1,
        f"queued={queued['queued']}",
    )
    for job in queued["jobs"]:
        wait_for_job(job["id"])

    after = analysis_for(project_id, media_id)
    check("no duplicate rows for the same analyzer version", len(after) == len(before))
    check(
        "analyzer versions are recorded",
        all(row["analyzer_version"] for row in after.values()),
    )


def main() -> int:
    print("VisionForge Phase 3 - media intelligence acceptance")
    print(f"API: {API}")

    try:
        health = httpx.get(f"{API}/health/ready", timeout=15).json()
    except httpx.HTTPError as exc:
        print(f"\nCannot reach the API at {API}: {exc}")
        return 2
    if health["status"] != "ok":
        print(f"\nAPI is not ready: {json.dumps(health, indent=2)}")
        return 2

    if not (FIXTURES / "image.jpg").exists():
        print("\nFixtures missing. Run: .\\scripts\\vf.ps1 test-integration")
        return 2

    project_id, media = scenario_pipeline()
    scenario_image(project_id, media["image.jpg"])
    scenario_video(project_id, media["video_1080.mp4"])
    scenario_audio(project_id, media["audio.mp3"])
    scenario_similarity(project_id, media)
    scenario_versioning(project_id, media["image.jpg"])

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All Phase 3 acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
