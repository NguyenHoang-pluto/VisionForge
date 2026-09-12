"""Phase 2 end-to-end acceptance test.

Exercises the whole system as a user would, against a live API, a live Celery
worker and live infrastructure:

    create project -> upload a folder of media -> MinIO -> ingest jobs ->
    ffprobe -> sha256 -> thumbnail -> 720p proxy -> SSE progress -> media library

then the four resilience scenarios Phase 2 has to prove:

    1. kill the worker mid-job, restart it, confirm the job completes
    2. upload a duplicate, confirm deduplication
    3. cancel a job, confirm cancellation
    4. upload a corrupt file, confirm it fails permanently without retrying

Run it with the API and a worker already up::

    python scripts/e2e_acceptance.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

API = os.environ.get("VF_API_URL", "http://127.0.0.1:8000")
FIXTURES = Path(__file__).resolve().parents[1] / "apps/api/tests/integration/fixtures"
TIMEOUT_S = 180

PASS, FAIL, INFO = "PASS", "FAIL", " ..."
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def step(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


@dataclass
class Uploaded:
    media_id: str
    job_id: str | None


# --------------------------------------------------------------------- helpers
def create_project(title: str) -> str:
    response = httpx.post(f"{API}/api/projects", json={"title": title}, timeout=30)
    response.raise_for_status()
    return str(response.json()["id"])


def upload(project_id: str, path: Path, *, filename: str | None = None) -> Uploaded:
    """Presign, PUT straight to storage, then tell the API it landed."""
    name = filename or path.name
    ticket = httpx.post(
        f"{API}/api/projects/{project_id}/media/upload-url",
        json={"filename": name, "size_bytes": path.stat().st_size},
        timeout=30,
    )
    ticket.raise_for_status()
    body = ticket.json()

    put = httpx.put(body["upload_url"], content=path.read_bytes(), timeout=120)
    put.raise_for_status()

    completed = httpx.post(
        f"{API}/api/projects/{project_id}/media/{body['media_id']}/complete", timeout=60
    )
    completed.raise_for_status()
    return Uploaded(media_id=body["media_id"], job_id=completed.json().get("job_id"))


def wait_for_job(job_id: str, *, want: set[str], timeout_s: int = TIMEOUT_S) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = httpx.get(f"{API}/api/jobs/{job_id}", timeout=30).json()
        if last["status"] in want:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} stuck in {last.get('status')} after {timeout_s}s")


def sse_first_frame(job_id: str) -> dict:
    with httpx.stream("GET", f"{API}/api/jobs/{job_id}/events", timeout=30) as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                return dict(json.loads(line[5:].strip()))
    raise AssertionError("no SSE frame")


def worker(action: str) -> subprocess.Popen | None:
    """Start or stop the Celery CPU worker."""
    if action == "start":
        return subprocess.Popen(  # noqa: S603 - argv array, local dev script
            [
                sys.executable, "-m", "celery",
                "-A", "visionforge.workers.cpu", "worker",
                "--pool=solo", "-Q", "cpu", "-l", "warning",
            ],
            cwd=str(Path(__file__).resolve().parents[1] / "apps/api"),
            env={**_env(), "PYTHONPATH": "src"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return None


def _env() -> dict[str, str]:
    return dict(os.environ)


# ----------------------------------------------------------------- the scenarios
def scenario_happy_path() -> None:
    step("1. Folder upload -> ingest -> media library")
    project_id = create_project("E2E Acceptance")
    print(f"{INFO} project {project_id}")

    folder = [
        FIXTURES / "image.jpg",
        FIXTURES / "image2.jpg",
        FIXTURES / "video_1080.mp4",
        FIXTURES / "audio.mp3",
    ]
    uploads = [upload(project_id, path) for path in folder]
    check("all files uploaded and queued", all(u.job_id for u in uploads))

    for item in uploads:
        assert item.job_id
        job = wait_for_job(item.job_id, want={"succeeded", "failed", "cancelled"})
        check(f"job {item.job_id[:8]} succeeded", job["status"] == "succeeded", job["status"])
        check(
            f"  progress reached 1.0 across {len(job['steps'])} real steps",
            job["progress"] == 1.0 and len(job["steps"]) == 6,
        )

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    check("media library lists 4 assets", listing["total"] == 4, str(listing["total"]))

    by_name = {m["original_filename"]: m for m in listing["items"]}

    video = by_name["video_1080.mp4"]
    check("video: ffprobe metadata", (video["width"], video["height"]) == (1920, 1080))
    check("video: duration and fps", bool(video["duration_ms"]) and bool(video["fps"]))
    check("video: sha256 computed", bool(video["sha256"]) and len(video["sha256"]) == 64)
    check("video: thumbnail generated", video["has_thumbnail"])
    check("video: 720p proxy generated", video["has_proxy"])

    image = by_name["image.jpg"]
    check("image: thumbnail generated", image["has_thumbnail"])
    check("image: no proxy (correctly skipped)", not image["has_proxy"])

    audio = by_name["audio.mp3"]
    check("audio: kind detected", audio["kind"] == "audio")
    check("audio: no thumbnail or proxy", not audio["has_thumbnail"] and not audio["has_proxy"])

    thumb = httpx.get(
        f"{API}/api/projects/{project_id}/media/{video['id']}/thumbnail", timeout=30
    ).json()
    check("presigned thumbnail URL serves bytes", httpx.get(thumb["url"], timeout=30).is_success)

    assert uploads[0].job_id
    frame = sse_first_frame(uploads[0].job_id)
    check("SSE replays last known state on reconnect", frame["status"] == "succeeded")


def scenario_worker_restart() -> None:
    step("2. Kill the worker mid-job, restart, confirm recovery")
    project_id = create_project("E2E Worker Restart")

    # Stop the worker first, so the job is queued with nothing to consume it.
    print(f"{INFO} stopping worker")
    subprocess.run(  # noqa: S603
        ["powershell", "-NoProfile", "-Command",
         "Get-Process python -EA SilentlyContinue | Where-Object {$_.CommandLine -like '*celery*'} "
         "| Stop-Process -Force -EA SilentlyContinue"],
        capture_output=True,
        timeout=60,
    )
    time.sleep(2)

    item = upload(project_id, FIXTURES / "video_1080.mp4")
    assert item.job_id
    queued = httpx.get(f"{API}/api/jobs/{item.job_id}", timeout=30).json()
    check("job survives with no worker running", queued["status"] in {"queued", "running"})
    check("  job state is in postgres, not the broker", queued["progress"] == 0.0)

    print(f"{INFO} restarting worker")
    process = worker("start")
    try:
        job = wait_for_job(item.job_id, want={"succeeded", "failed"})
        check("job completes after worker restart", job["status"] == "succeeded", job["status"])
    finally:
        if process:
            process.terminate()
            process.wait(timeout=30)


def scenario_duplicate(project_id: str) -> None:
    step("3. Duplicate upload -> deduplication")
    first = upload(project_id, FIXTURES / "image.jpg")
    assert first.job_id
    wait_for_job(first.job_id, want={"succeeded", "failed"})

    second = upload(project_id, FIXTURES / "image.jpg", filename="same-bytes-copy.jpg")
    assert second.job_id
    job = wait_for_job(second.job_id, want={"succeeded", "failed"})

    check("duplicate job succeeds (not an error)", job["status"] == "succeeded", job["status"])
    check("job result flags the duplicate", bool(job["result"].get("duplicate")))
    check(
        "  points at the original asset",
        job["result"].get("duplicate_of") == first.media_id,
    )

    listing = httpx.get(f"{API}/api/projects/{project_id}/media", timeout=30).json()
    check("library holds one asset, not two", listing["total"] == 1, str(listing["total"]))


def scenario_cancel() -> None:
    step("4. Cancel a job")
    project_id = create_project("E2E Cancel")

    subprocess.run(  # noqa: S603 - pause the worker so the job stays queued
        ["powershell", "-NoProfile", "-Command",
         "Get-Process python -EA SilentlyContinue | Where-Object {$_.CommandLine -like '*celery*'} "
         "| Stop-Process -Force -EA SilentlyContinue"],
        capture_output=True,
        timeout=60,
    )
    time.sleep(2)

    item = upload(project_id, FIXTURES / "video_1080.mp4")
    assert item.job_id

    cancelled = httpx.post(f"{API}/api/jobs/{item.job_id}/cancel", timeout=30).json()
    check("queued job cancels immediately", cancelled["status"] == "cancelled")
    check("  cancel flag is recorded", cancelled["cancel_requested"] is True)

    process = worker("start")
    try:
        time.sleep(6)
        final = httpx.get(f"{API}/api/jobs/{item.job_id}", timeout=30).json()
        check("worker does not process a cancelled job", final["status"] == "cancelled")

        media = httpx.get(
            f"{API}/api/projects/{project_id}/media/{item.media_id}", timeout=30
        ).json()
        check("  media never became ready", media["status"] != "ready", media["status"])
    finally:
        if process:
            process.terminate()
            process.wait(timeout=30)


def scenario_corrupt(project_id: str) -> None:
    step("5. Corrupt file -> permanent failure, no retries")
    corrupt = FIXTURES / "corrupt.jpg"
    corrupt.write_bytes(b"MZ\x90\x00" + b"\x00" * 512)

    item = upload(project_id, corrupt)
    assert item.job_id
    job = wait_for_job(item.job_id, want={"succeeded", "failed"})

    check("corrupt media fails", job["status"] == "failed", job["status"])
    check("  fails on the first attempt (no wasted retries)", job["attempts"] == 1)
    check("  error is marked non-retryable", job["error"].get("retryable") is False)

    failed_steps = [s["name"] for s in job["steps"] if s["status"] == "failed"]
    check("  rejected at VALIDATE, before any transcode", failed_steps == ["VALIDATE"])

    media = httpx.get(
        f"{API}/api/projects/{project_id}/media/{item.media_id}", timeout=30
    ).json()
    check("  media marked failed", media["status"] == "failed", media["status"])


def main() -> int:
    print("VisionForge Phase 2 - end-to-end acceptance")
    print(f"API: {API}")

    try:
        health = httpx.get(f"{API}/health/ready", timeout=15).json()
    except httpx.HTTPError as exc:
        print(f"\nCannot reach the API at {API}: {exc}")
        print("Start it with:  .\\scripts\\vf.ps1 api")
        return 2
    if health["status"] != "ok":
        print(f"\nAPI is not ready: {json.dumps(health, indent=2)}")
        return 2

    if not (FIXTURES / "image.jpg").exists():
        print(f"\nFixtures missing. Generate them with:  .\\scripts\\vf.ps1 test-integration")
        return 2

    worker_process = worker("start")
    time.sleep(6)
    try:
        scenario_happy_path()

        dedupe_project = create_project("E2E Deduplication")
        scenario_duplicate(dedupe_project)

        failure_project = create_project("E2E Corrupt Media")
        scenario_corrupt(failure_project)
    finally:
        if worker_process:
            worker_process.terminate()
            worker_process.wait(timeout=30)

    scenario_worker_restart()
    scenario_cancel()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED - {len(_failures)} check(s) did not pass:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("All acceptance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
