"""Celery publisher: the only place that knows task names."""

from __future__ import annotations

from uuid import UUID

from visionforge.domain.jobs import JobType
from visionforge.infra.queue.celery_app import QUEUE_CPU, QUEUE_GPU, celery_app

#: Job type -> (task name, queue). Media ingest is CPU work: ffprobe, hashing
#: and FFmpeg transcodes, no GPU.
TASK_ROUTES: dict[JobType, tuple[str, str]] = {
    JobType.MEDIA_INGEST: ("visionforge.media_ingest", QUEUE_CPU),
    JobType.MEDIA_ANALYZE_CPU: ("visionforge.media_analyze_cpu", QUEUE_CPU),
    # Model inference goes to the gpu queue, whose worker holds the single
    # concurrency slot that serialises access to the 4 GiB card.
    JobType.MEDIA_ANALYZE_GPU: ("visionforge.media_analyze_gpu", QUEUE_GPU),
}


class CeleryTaskPublisher:
    def publish(self, job_type: JobType, job_id: UUID) -> str | None:
        task_name, queue = TASK_ROUTES[job_type]
        result = celery_app.send_task(task_name, args=[str(job_id)], queue=queue)
        return str(result.id) if result.id else None
