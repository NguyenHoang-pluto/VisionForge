"""Planner determinism, timeline compilation, render spec and argv generation.

The whole deterministic chain, exercised without touching a file or spawning a
process. The argv compiler being pure is what makes the exact command
assertable, rather than something inferred from whether a render happened to
work.
"""

from __future__ import annotations

import itertools
import uuid

import pytest

from visionforge.domain.editplan import (
    MAX_SEGMENT_MS,
    MIN_SEGMENT_MS,
    AspectRatio,
    AudioMode,
    FitMode,
    MediaFact,
    validate_plan,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind
from visionforge.domain.planner import (
    ClipOrder,
    NoUsableMediaError,
    PlanRequest,
    RulesEnginePlanner,
)
from visionforge.domain.selection import Candidate
from visionforge.domain.timeline import (
    TrackKind,
    VideoCodec,
    build_render_spec,
    compile_timeline,
)
from visionforge.infra.ffmpeg.compiler import (
    build_filter_graph,
    compile_render_argv,
    parse_progress_line,
    progress_fraction,
    scale_filter,
)

PROJECT = ProjectId(uuid.uuid4())


def candidate(sequence: int, *, blur: float = 300.0, duration: int = 8000) -> Candidate:
    return Candidate(
        media_id=MediaId(uuid.uuid4()),
        kind=MediaKind.VIDEO,
        is_ready=True,
        duration_ms=duration,
        width=1920,
        height=1080,
        blur_score=blur,
        contrast=60.0,
        mean_luminance=128.0,
        clipped_ratio=0.01,
        phash=f"{(sequence * 0x1111111111111111) & ((1 << 64) - 1):016x}",
        sequence=sequence,
    )


def pool(count: int) -> list[Candidate]:
    # Descending sharpness so the score ordering is known up front.
    return [candidate(i, blur=500.0 - i * 30) for i in range(count)]


def facts_for(candidates: list[Candidate]) -> dict[MediaId, MediaFact]:
    return {
        c.media_id: MediaFact(
            media_id=c.media_id,
            project_id=PROJECT,
            is_renderable=True,
            duration_ms=c.duration_ms,
            width=c.width,
            height=c.height,
        )
        for c in candidates
    }


# ------------------------------------------------------------------- planner
class TestRulesEnginePlanner:
    def test_produces_a_valid_plan(self) -> None:
        candidates = pool(8)
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT), candidates)

        assert validate_plan(outcome.plan, facts_for(candidates)) == []

    def test_respects_the_clip_budget(self) -> None:
        candidates = pool(10)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=5), candidates
        )
        assert len(outcome.plan.segments) == 5

    def test_hits_the_target_duration_when_the_material_allows(self) -> None:
        candidates = pool(5)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=25_000, max_clips=5),
            candidates,
        )
        # 25000 / 5 = 5000 per clip, and every source is 8 s so nothing clamps.
        assert outcome.plan.total_duration_ms == 25_000

    def test_is_deterministic(self) -> None:
        """Two runs over the same folder must produce the same edit."""
        candidates = pool(10)
        request = PlanRequest(project_id=PROJECT, max_clips=5)

        first = RulesEnginePlanner().plan(request, candidates).plan
        second = RulesEnginePlanner().plan(request, candidates).plan

        assert first.as_payload()["segments"] == second.as_payload()["segments"]

    def test_orders_by_score_by_default(self) -> None:
        candidates = pool(5)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=3, order=ClipOrder.SCORE_DESC),
            candidates,
        )
        scores = {s.candidate.media_id: s.score for s in outcome.selection.selected}
        chosen = [scores[s.media_id] for s in outcome.plan.ordered_segments]
        assert chosen == sorted(chosen, reverse=True)

    def test_sequence_order_preserves_upload_order(self) -> None:
        """For a day's shooting that is usually chronological, and usually right."""
        candidates = pool(5)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=5, order=ClipOrder.SEQUENCE),
            candidates,
        )
        sequences = {c.media_id: c.sequence for c in candidates}
        chosen = [sequences[s.media_id] for s in outcome.plan.ordered_segments]
        assert chosen == sorted(chosen)

    def test_trims_from_the_centre(self) -> None:
        """The opening of a handheld clip is disproportionately unusable."""
        candidates = [candidate(0, duration=10_000)]
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=4000, max_clips=1, min_clips=1),
            candidates,
        )
        segment = outcome.plan.segments[0]
        assert segment.source_in_ms == 3000
        assert segment.source_out_ms == 7000

    def test_a_short_source_is_used_whole_rather_than_over_trimmed(self) -> None:
        candidates = [candidate(0, duration=2000)]
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=10_000, max_clips=1, min_clips=1),
            candidates,
        )
        segment = outcome.plan.segments[0]
        assert (segment.source_in_ms, segment.source_out_ms) == (0, 2000)

    def test_per_clip_duration_is_clamped_to_the_segment_bounds(self) -> None:
        candidates = pool(2)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=120_000, max_clips=2),
            candidates,
        )
        for segment in outcome.plan.segments:
            assert MIN_SEGMENT_MS <= segment.duration_ms <= MAX_SEGMENT_MS

    def test_refuses_when_too_little_survives_selection(self) -> None:
        unusable = [candidate(i, blur=2.0) for i in range(5)]
        with pytest.raises(NoUsableMediaError, match="usable clip"):
            RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT, min_clips=2), unusable)

    def test_segment_orders_are_contiguous_from_zero(self) -> None:
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT, max_clips=4), pool(6))
        assert [s.order for s in outcome.plan.ordered_segments] == [0, 1, 2, 3]

    def test_the_plan_records_how_it_was_made(self) -> None:
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT), pool(5))
        metadata = outcome.plan.metadata

        assert outcome.plan.planner == "rules-engine"
        assert metadata["order"] == ClipOrder.SCORE_DESC.value
        assert "weights" in metadata
        assert metadata["considered"] == 5

    def test_the_selection_explains_what_was_dropped(self) -> None:
        candidates = [*pool(4), candidate(99, blur=2.0)]
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=4), candidates
        )
        assert any(r.reason.value == "too_blurry" for r in outcome.selection.rejected)


class TestPlanRequestValidation:
    def test_max_clips_cannot_be_below_min(self) -> None:
        with pytest.raises(ValueError, match="max_clips"):
            PlanRequest(project_id=PROJECT, min_clips=5, max_clips=2)

    def test_target_duration_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="target_duration_ms"):
            PlanRequest(project_id=PROJECT, target_duration_ms=0)


# ------------------------------------------------------------------- timeline
class TestTimelineCompilation:
    def test_clips_are_butt_joined(self) -> None:
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT, max_clips=3), pool(3))
        clips = compile_timeline(outcome.plan).video_track.clips

        for previous, current in itertools.pairwise(clips):
            assert current.timeline_start_ms == previous.timeline_end_ms

    def test_timeline_duration_matches_the_plan(self) -> None:
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT, max_clips=4), pool(4))
        assert compile_timeline(outcome.plan).duration_ms == outcome.plan.total_duration_ms

    def test_source_and_timeline_coordinates_stay_distinct(self) -> None:
        """Conflating them is how editors end up with drift."""
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, target_duration_ms=6000, max_clips=2), pool(2)
        )
        second = compile_timeline(outcome.plan).video_track.clips[1]

        assert second.timeline_start_ms != second.source_in_ms

    def test_no_audio_track_when_audio_is_off(self) -> None:
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, audio=AudioMode.NONE), pool(3)
        )
        kinds = {t.kind for t in compile_timeline(outcome.plan).tracks}
        assert kinds == {TrackKind.VIDEO}

    def test_an_audio_track_appears_when_requested(self) -> None:
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, audio=AudioMode.SOURCE), pool(3)
        )
        kinds = {t.kind for t in compile_timeline(outcome.plan).tracks}
        assert kinds == {TrackKind.VIDEO, TrackKind.AUDIO}

    def test_the_timeline_carries_no_ffmpeg_concepts(self) -> None:
        """FFmpeg-free by construction, so a manual editor can mutate it later."""
        outcome = RulesEnginePlanner().plan(PlanRequest(project_id=PROJECT), pool(3))
        payload = compile_timeline(outcome.plan).as_payload()
        serialised = str(payload).lower()

        for forbidden in ("libx264", "filter", "crf", "ffmpeg", "/", "codec"):
            assert forbidden not in serialised


# ---------------------------------------------------------------- render spec
class TestRenderSpec:
    def _spec(self, count: int = 3, **kwargs: object):  # type: ignore[no-untyped-def]
        candidates = pool(count)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=count, min_clips=1, **kwargs),  # type: ignore[arg-type]
            candidates,
        )
        timeline = compile_timeline(outcome.plan)
        paths = {c.media_id: f"/w/{c.sequence}.mp4" for c in candidates}
        return build_render_spec(timeline, local_paths=paths, output_path="/w/out.mp4")

    def test_one_input_per_distinct_media(self) -> None:
        spec = self._spec(3)
        assert len(spec.inputs) == 3
        assert len(spec.segments) == 3

    def test_inputs_are_indexed_from_zero(self) -> None:
        spec = self._spec(3)
        assert [i.index for i in spec.inputs] == [0, 1, 2]

    def test_a_missing_local_path_is_refused(self) -> None:
        """Rather than producing a spec FFmpeg would fail on obscurely."""
        candidates = pool(2)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(project_id=PROJECT, max_clips=2), candidates
        )
        timeline = compile_timeline(outcome.plan)

        with pytest.raises(ValueError, match="no local path"):
            build_render_spec(timeline, local_paths={}, output_path="/w/out.mp4")

    def test_defaults_are_sane_for_this_hardware(self) -> None:
        spec = self._spec(2)
        assert spec.video_codec is VideoCodec.H264
        assert spec.preset == "veryfast"
        assert spec.pixel_format == "yuv420p"

    def test_the_payload_omits_local_paths(self) -> None:
        """A worker-local path is meaningless to a client and not for handing out."""
        payload = self._spec(2).as_payload()
        assert "/w/" not in str(payload)
        assert "output_path" not in payload


# ------------------------------------------------------------- filter graph
class TestFilterGraph:
    def test_cover_crops_and_contain_pads(self) -> None:
        cover = scale_filter(1280, 720, FitMode.COVER)
        contain = scale_filter(1280, 720, FitMode.CONTAIN)

        assert "crop=1280:720" in cover and "pad=" not in cover
        assert "pad=1280:720" in contain and "crop=" not in contain

    def test_both_fits_set_the_sample_aspect_ratio(self) -> None:
        """Without setsar, a non-square SAR source renders stretched."""
        assert "setsar=1" in scale_filter(1280, 720, FitMode.COVER)
        assert "setsar=1" in scale_filter(1280, 720, FitMode.CONTAIN)

    def test_each_segment_is_trimmed_rebased_scaled_and_rate_forced(self) -> None:
        """concat requires uniform inputs; a real folder will not provide them."""
        spec = TestRenderSpec()._spec(2)
        graph = build_filter_graph(spec)

        assert graph.count("trim=") == 2
        assert graph.count("setpts=PTS-STARTPTS") == 2
        assert graph.count("fps=30") == 2
        assert graph.count("format=yuv420p") == 2

    def test_the_graph_ends_in_a_concat(self) -> None:
        spec = TestRenderSpec()._spec(3)
        assert "concat=n=3:v=1:a=0[vout]" in build_filter_graph(spec)

    def test_audio_is_trimmed_and_resampled_in_parallel(self) -> None:
        spec = TestRenderSpec()._spec(2, audio=AudioMode.SOURCE)
        graph = build_filter_graph(spec)

        # Two per-segment trims, plus the one that cuts the finished audio to
        # the video's length.
        assert graph.count("atrim=") == 3
        assert "aformat=sample_rates=48000" in graph

    def test_video_and_audio_are_concatenated_separately(self) -> None:
        """Phase 7 split what Phase 4 did in one filter.

        A single ``concat=v=1:a=1`` emits both streams at once, which leaves
        nowhere to mix a music bed in afterwards. Splitting them changes no
        output when there is no music -- same segments, same order, same rate --
        and is what makes the mix expressible when there is.
        """
        spec = TestRenderSpec()._spec(2, audio=AudioMode.SOURCE)
        graph = build_filter_graph(spec)

        assert "concat=n=2:v=1:a=0[vout]" in graph
        assert "concat=n=2:v=0:a=1[asrc]" in graph
        assert "v=1:a=1" not in graph

    def test_the_video_graph_is_identical_with_and_without_source_audio(self) -> None:
        """The regression that matters: adding audio must not move a frame."""
        silent = build_filter_graph(TestRenderSpec()._spec(3))
        with_audio = build_filter_graph(TestRenderSpec()._spec(3, audio=AudioMode.SOURCE))

        video_only = [p for p in with_audio.split(";") if "[v" in p or "[vout]" in p]
        assert silent.split(";") == video_only

    def test_trim_times_are_seconds_with_millisecond_precision(self) -> None:
        """FFmpeg takes seconds; explicit formatting avoids scientific notation."""
        import re

        graph = build_filter_graph(TestRenderSpec()._spec(1))
        starts = re.findall(r"trim=start=([0-9.]+):end=([0-9.]+)", graph)

        assert starts
        for start, end in starts:
            assert len(start.split(".")[1]) == 3
            assert len(end.split(".")[1]) == 3


# ---------------------------------------------------------------------- argv
class TestArgvCompiler:
    def test_returns_a_list_never_a_string(self) -> None:
        """The structural reason no shell injection is possible."""
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert isinstance(argv, list)
        assert all(isinstance(arg, str) for arg in argv)

    def test_does_not_include_the_executable(self) -> None:
        """The runner resolves the binary, so this cannot be pointed elsewhere."""
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert "ffmpeg" not in argv[0]

    def test_a_hostile_filename_stays_one_argument(self) -> None:
        """argv, not a shell string: metacharacters are data, not syntax."""
        from visionforge.domain.timeline import RenderInput, RenderSegment, RenderSpec

        hostile = "/w/clip; rm -rf ~.mp4"
        spec = RenderSpec(
            inputs=(RenderInput(media_id=MediaId(uuid.uuid4()), index=0, local_path=hostile),),
            segments=(RenderSegment(input_index=0, source_in_ms=0, source_out_ms=3000),),
            width=1280,
            height=720,
            fps=30,
            fit=FitMode.COVER,
            output_path="/w/out.mp4",
        )
        argv = compile_render_argv(spec)

        # The path survives intact as exactly one element -- never split on the
        # semicolon, never re-quoted, never merged with a neighbour. argv is a
        # list, so ';' is data. (The filtergraph argument also contains ';', but
        # that is FFmpeg's chain separator, not shell syntax.)
        assert argv.count(hostile) == 1
        assert argv[argv.index("-i") + 1] == hostile
        assert not any(arg.startswith("rm ") or arg == "rm" for arg in argv)

    def test_every_input_is_passed_with_its_own_flag(self) -> None:
        argv = compile_render_argv(TestRenderSpec()._spec(3))
        assert argv.count("-i") == 3

    def test_video_is_mapped_from_the_concat_output(self) -> None:
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert argv[argv.index("-map") + 1] == "[vout]"

    def test_audio_is_disabled_when_not_requested(self) -> None:
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert "-an" in argv
        assert "[aout]" not in argv

    def test_audio_is_mapped_and_encoded_when_requested(self) -> None:
        argv = compile_render_argv(TestRenderSpec()._spec(2, audio=AudioMode.SOURCE))
        assert "[aout]" in argv
        assert "aac" in argv
        assert "-an" not in argv

    def test_encoder_settings_come_from_the_spec(self) -> None:
        spec = TestRenderSpec()._spec(2)
        argv = compile_render_argv(spec)

        assert argv[argv.index("-crf") + 1] == str(spec.crf)
        assert argv[argv.index("-preset") + 1] == spec.preset
        assert argv[argv.index("-pix_fmt") + 1] == spec.pixel_format

    def test_faststart_is_requested(self) -> None:
        """So playback can begin before the whole file has downloaded."""
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert argv[argv.index("-movflags") + 1] == "+faststart"

    def test_progress_reporting_is_enabled(self) -> None:
        argv = compile_render_argv(TestRenderSpec()._spec(2))
        assert "-progress" in argv and "pipe:1" in argv

    def test_the_output_path_is_the_final_argument(self) -> None:
        spec = TestRenderSpec()._spec(2)
        assert compile_render_argv(spec)[-1] == spec.output_path

    def test_is_pure_and_deterministic(self) -> None:
        spec = TestRenderSpec()._spec(3)
        assert compile_render_argv(spec) == compile_render_argv(spec)

    def test_an_empty_spec_is_refused(self) -> None:
        from visionforge.domain.timeline import RenderSpec

        empty = RenderSpec(
            inputs=(),
            segments=(),
            width=1280,
            height=720,
            fps=30,
            fit=FitMode.COVER,
            output_path="/w/out.mp4",
        )
        with pytest.raises(ValueError, match="no inputs"):
            compile_render_argv(empty)


# ------------------------------------------------------------------ progress
class TestProgressParsing:
    def test_parses_a_key_value_line(self) -> None:
        assert parse_progress_line("out_time_us=1234567") == ("out_time_us", "1234567")

    @pytest.mark.parametrize("line", ["", "no equals here", "=", "key=", "=value"])
    def test_malformed_lines_are_ignored_not_fatal(self, line: str) -> None:
        """A bad line must not be able to break progress reporting."""
        assert parse_progress_line(line) is None

    def test_fraction_from_elapsed_microseconds(self) -> None:
        assert progress_fraction(5_000_000, 10_000) == pytest.approx(0.5)

    def test_fraction_is_clamped_past_the_end(self) -> None:
        """FFmpeg occasionally reports a timestamp beyond the duration."""
        assert progress_fraction(20_000_000, 10_000) == 1.0

    def test_unknown_duration_reports_zero_rather_than_dividing_by_zero(self) -> None:
        assert progress_fraction(5_000_000, 0) == 0.0


# ---------------------------------------------------- aspect ratio geometry
class TestAspectRatios:
    @pytest.mark.parametrize(
        ("ratio", "width", "height"),
        [
            (AspectRatio.LANDSCAPE_16_9, 1280, 720),
            (AspectRatio.PORTRAIT_9_16, 720, 1280),
            (AspectRatio.SQUARE_1_1, 720, 720),
        ],
    )
    def test_each_preset_plans_and_compiles(
        self, ratio: AspectRatio, width: int, height: int
    ) -> None:
        candidates = pool(3)
        outcome = RulesEnginePlanner().plan(
            PlanRequest(
                project_id=PROJECT,
                max_clips=3,
                aspect_ratio=ratio,
                width=width,
                height=height,
            ),
            candidates,
        )
        assert validate_plan(outcome.plan, facts_for(candidates)) == []

        timeline = compile_timeline(outcome.plan)
        spec = build_render_spec(
            timeline,
            local_paths={c.media_id: f"/w/{c.sequence}.mp4" for c in candidates},
            output_path="/w/out.mp4",
        )
        assert f"scale={width}:{height}" in build_filter_graph(spec)
