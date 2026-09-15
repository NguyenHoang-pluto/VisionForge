"""Every declared job step has a handler, and every handler is declared.

These are two different lists in two different layers, and they have to agree:

- ``domain.jobs.JOB_STEP_PLANS`` is the *contract*. A job's ``job_steps`` rows
  are created from it, so a step missing here never runs, however correctly it
  is registered in the worker.
- ``workers.*.STEPS`` is the *dispatch table*. A step declared in the domain
  with no handler here fails the job at that step.

Phase 7 shipped a bug in exactly the first direction: `BEATS` was added to the
worker's dispatch table and not to the domain's tuple, so beat detection was
registered, importable, unit-tested — and never executed by a single real job.
The acceptance run caught it. This catches it next time, in a second rather
than in a fifteen-minute pipeline.
"""

from __future__ import annotations

import pytest

from visionforge.domain.jobs import JOB_STEP_PLANS, JobType
from visionforge.workers import media_analysis, media_ingest, video_render

#: Which dispatch table serves each job type.
DISPATCH: dict[JobType, dict[str, object]] = {
    JobType.MEDIA_INGEST: media_ingest.STEPS,
    JobType.MEDIA_ANALYZE_CPU: media_analysis.CPU_STEPS,
    JobType.MEDIA_ANALYZE_GPU: media_analysis.GPU_STEPS,
    JobType.RENDER_VIDEO: video_render.STEPS,
}


class TestStepContract:
    def test_every_job_type_has_a_dispatch_table(self) -> None:
        assert set(DISPATCH) == set(JOB_STEP_PLANS)

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_every_declared_step_has_a_handler(self, job_type: JobType) -> None:
        """The Phase 7 bug, inverted: a step with no handler fails its job."""
        declared = set(JOB_STEP_PLANS[job_type])
        handled = set(DISPATCH[job_type])
        assert (
            declared <= handled
        ), f"{job_type.value} declares steps with no handler: {declared - handled}"

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_every_handler_is_declared(self, job_type: JobType) -> None:
        """The Phase 7 bug as it actually happened: a handler nothing runs."""
        declared = set(JOB_STEP_PLANS[job_type])
        handled = set(DISPATCH[job_type])
        assert (
            handled <= declared
        ), f"{job_type.value} has handlers no job runs: {handled - declared}"

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_the_two_lists_are_the_same_set(self, job_type: JobType) -> None:
        assert set(JOB_STEP_PLANS[job_type]) == set(DISPATCH[job_type])

    @pytest.mark.parametrize("job_type", list(JobType))
    def test_every_handler_is_callable(self, job_type: JobType) -> None:
        for name, handler in DISPATCH[job_type].items():
            assert callable(handler), f"{job_type.value}.{name} is not callable"

    def test_cpu_analysis_runs_beat_detection(self) -> None:
        """Named explicitly, because "it is in the dict" was true and useless."""
        assert "BEATS" in JOB_STEP_PLANS[JobType.MEDIA_ANALYZE_CPU]
        assert "BEATS" in media_analysis.CPU_STEPS

    def test_steps_start_by_resolving_and_end_by_finalizing(self) -> None:
        """Order matters: nothing can run before the input is local, and the
        scratch directory is removed by FINALIZE."""
        for job_type, steps in JOB_STEP_PLANS.items():
            assert steps[-1] == "FINALIZE", job_type.value
            if job_type is not JobType.MEDIA_INGEST:
                assert steps[0] in {"RESOLVE", "PREPARE"}, job_type.value
