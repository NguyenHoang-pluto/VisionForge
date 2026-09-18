"""Beat-aware clip placement.

The claim under test is narrow and checkable: when a trustworthy grid exists,
every cut in the finished plan lands on a beat -- and when one does not, the
planner produces exactly the edit it produced before this feature existed.

The second half matters more than the first. Graceful degradation is the
requirement, and the ways to fail it are all silent: analysis that never ran,
analysis that ran on a podcast, a tempo so slow that quantising would double
every clip. Each gets a test.
"""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.beats import MIN_BEAT_CONFIDENCE, BeatGrid
from visionforge.domain.editplan import MAX_SEGMENT_MS, MIN_SEGMENT_MS
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind
from visionforge.domain.planner import ClipOrder, PlanRequest, RulesEnginePlanner
from visionforge.domain.selection import Candidate
from visionforge.domain.style import EditStyle

PROJECT = ProjectId(uuid.uuid4())
TRACK = MediaId(uuid.uuid4())


def grid(bpm: float = 120.0, confidence: float = 0.9, count: int = 200, start: int = 0) -> BeatGrid:
    period = 60_000.0 / bpm
    return BeatGrid(
        bpm=bpm,
        confidence=confidence,
        beats_ms=tuple(round(start + i * period) for i in range(count)),
        source_duration_ms=round(start + count * period),
    )


def candidates(count: int = 5, duration_ms: int = 20_000) -> list[Candidate]:
    """Clips good enough to survive selection, distinguishable by hash."""
    return [
        Candidate(
            media_id=MediaId(uuid.uuid4()),
            kind=MediaKind.VIDEO,
            is_ready=True,
            duration_ms=duration_ms,
            width=1920,
            height=1080,
            blur_score=400.0,
            contrast=50.0,
            mean_luminance=128.0,
            clipped_ratio=0.01,
            # Distinct hashes, and *far apart*: the selector drops anything
            # within PHASH_DUPLICATE_MAX_DISTANCE (5) bits of a clip it already
            # took. Sequential counters differ by one bit, so four of five
            # candidates were being deduplicated away and every test below was
            # quietly planning a single clip. These are 16 bits apart.
            phash=f"{(i * 0x1111111111111111) & 0xFFFFFFFFFFFFFFFF:016x}",
            sequence=i,
        )
        for i in range(count)
    ]


def request(**overrides: object) -> PlanRequest:
    values: dict = {
        "project_id": PROJECT,
        "target_duration_ms": 20_000,
        "max_clips": 4,
        "min_clips": 1,
        "order": ClipOrder.SEQUENCE,
    }
    values.update(overrides)
    return PlanRequest(**values)  # type: ignore[arg-type]


def cut_positions(plan) -> list[int]:  # type: ignore[no-untyped-def]
    """Where each cut falls on the finished timeline, cumulative from zero."""
    positions, cursor = [], 0
    for segment in plan.ordered_segments:
        cursor += segment.duration_ms
        positions.append(cursor)
    return positions


# ------------------------------------------------------------------ quantising
class TestBeatQuantisedCuts:
    def test_every_cut_lands_on_a_beat(self) -> None:
        """The whole claim, checked end to end on a finished plan."""
        outcome = RulesEnginePlanner().plan(
            request(beats=grid(120.0), beat_sync=True), candidates()
        )
        period = 500.0
        for position in cut_positions(outcome.plan):
            assert position % period == pytest.approx(0.0, abs=1.0)

    def test_clip_length_is_a_whole_number_of_beats(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(target_duration_ms=20_000, max_clips=4, beats=grid(120.0), beat_sync=True),
            candidates(),
        )
        # 20 s over 4 clips is 5 s each, which at 120 BPM is exactly 10 beats.
        for segment in outcome.plan.segments:
            assert segment.duration_ms == 5_000

    def test_an_awkward_target_is_rounded_to_the_nearest_beat(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(target_duration_ms=17_000, max_clips=4, beats=grid(120.0), beat_sync=True),
            candidates(),
        )
        # 4250 ms per clip rounds to 8.5 -> 8 or 9 beats; either is whole.
        span = outcome.plan.segments[0].duration_ms
        assert span % 500 == 0

    @pytest.mark.parametrize("bpm", [90.0, 128.0, 140.0])
    def test_it_holds_at_other_tempi(self, bpm: float) -> None:
        outcome = RulesEnginePlanner().plan(request(beats=grid(bpm), beat_sync=True), candidates())
        period = 60_000.0 / bpm
        for position in cut_positions(outcome.plan):
            remainder = position % period
            assert min(remainder, period - remainder) <= 1.0

    def test_the_plan_records_what_it_synced_to(self) -> None:
        """A beat-synced edit that cannot say which tempo is not reviewable."""
        outcome = RulesEnginePlanner().plan(
            request(beats=grid(128.0), beat_sync=True), candidates()
        )
        sync = outcome.plan.metadata["beat_sync"]
        assert sync["applied"] is True
        assert sync["bpm"] == pytest.approx(128.0, abs=0.01)
        assert sync["beats_per_clip"] == pytest.approx(round(sync["beats_per_clip"]), abs=1e-6)

    def test_no_gaps_or_overlaps_are_introduced(self) -> None:
        """Orders stay contiguous from zero; position is implied by order, so
        there is no field in which a gap could be expressed."""
        outcome = RulesEnginePlanner().plan(
            request(beats=grid(120.0), beat_sync=True), candidates()
        )
        orders = [segment.order for segment in outcome.plan.ordered_segments]
        assert orders == list(range(len(orders)))


# ------------------------------------------------------------- falling back
class TestGracefulFallback:
    def _unsynced(self, **overrides: object):  # type: ignore[no-untyped-def]
        return RulesEnginePlanner().plan(request(**overrides), candidates())

    def test_no_music_plans_exactly_as_before(self) -> None:
        baseline = self._unsynced()
        assert baseline.plan.metadata["beat_sync"]["applied"] is False
        assert baseline.plan.metadata["beat_sync"]["reason"] == "not_requested"

    def test_beats_without_beat_sync_do_not_move_a_cut(self) -> None:
        """Adding a bed and re-timing the edit are separate decisions."""
        plain = self._unsynced()
        with_beats = self._unsynced(beats=grid(137.0), beat_sync=False)
        assert [s.duration_ms for s in with_beats.plan.segments] == [
            s.duration_ms for s in plain.plan.segments
        ]

    def test_missing_analysis_falls_back(self) -> None:
        outcome = self._unsynced(beat_sync=True, beats=None)
        assert outcome.plan.metadata["beat_sync"]["reason"] == "no_beats"

    def test_low_confidence_falls_back(self) -> None:
        """A spoken-word bed must not silently re-time every cut."""
        outcome = self._unsynced(beat_sync=True, beats=grid(confidence=MIN_BEAT_CONFIDENCE - 0.05))
        sync = outcome.plan.metadata["beat_sync"]
        assert sync["applied"] is False
        assert sync["reason"] == "low_confidence"
        assert sync["confidence"] == pytest.approx(MIN_BEAT_CONFIDENCE - 0.05)

    def test_the_fallback_produces_the_unsynced_durations(self) -> None:
        plain = self._unsynced()
        degraded = self._unsynced(beat_sync=True, beats=grid(confidence=0.05))
        assert [s.duration_ms for s in degraded.plan.segments] == [
            s.duration_ms for s in plain.plan.segments
        ]

    def test_a_tempo_outside_the_searched_range_falls_back(self) -> None:
        """Out of range means detection clamped rather than measured."""
        outcome = self._unsynced(beat_sync=True, beats=grid(bpm=30.0))
        assert outcome.plan.metadata["beat_sync"]["applied"] is False


# ----------------------------------------------------------------- the bounds
class TestBoundsWin:
    def test_quantising_never_exceeds_the_segment_ceiling(self) -> None:
        """A very slow grid must not push a clip past what is renderable."""
        outcome = RulesEnginePlanner().plan(
            request(target_duration_ms=120_000, max_clips=2, beats=grid(61.0), beat_sync=True),
            candidates(duration_ms=60_000),
        )
        for segment in outcome.plan.segments:
            assert MIN_SEGMENT_MS <= segment.duration_ms <= MAX_SEGMENT_MS

    def test_quantising_never_goes_below_the_segment_floor(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(target_duration_ms=1_200, max_clips=4, beats=grid(200.0), beat_sync=True),
            candidates(),
        )
        for segment in outcome.plan.segments:
            assert segment.duration_ms >= MIN_SEGMENT_MS

    def test_a_style_still_constrains_pacing(self) -> None:
        """Style bounds apply before the grid; the plan bounds apply after both,
        so neither a style nor a tempo can widen what is renderable."""
        outcome = RulesEnginePlanner().plan(
            request(
                style=EditStyle.CINEMATIC,
                target_duration_ms=40_000,
                max_clips=4,
                beats=grid(120.0),
                beat_sync=True,
            ),
            candidates(duration_ms=40_000),
        )
        for segment in outcome.plan.segments:
            assert MIN_SEGMENT_MS <= segment.duration_ms <= MAX_SEGMENT_MS

    def test_a_trim_never_runs_past_the_source(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(target_duration_ms=40_000, max_clips=2, beats=grid(120.0), beat_sync=True),
            candidates(count=2, duration_ms=6_000),
        )
        for segment in outcome.plan.segments:
            assert segment.source_out_ms <= 6_000


# ------------------------------------------------------------------ the cue
class TestMusicCue:
    def test_no_music_id_means_no_cue(self) -> None:
        outcome = RulesEnginePlanner().plan(request(), candidates())
        assert outcome.plan.music is None

    def test_a_cue_is_written_when_a_track_is_chosen(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(music_media_id=TRACK, music_duration_ms=180_000), candidates()
        )
        cue = outcome.plan.music
        assert cue is not None
        assert cue.media_id == TRACK
        assert cue.timeline_start_ms == 0

    def test_the_cue_starts_on_the_first_beat_when_synced(self) -> None:
        """So the downbeat coincides with the first cut rather than the head of
        the file, which is where a pickup bar would otherwise put it."""
        outcome = RulesEnginePlanner().plan(
            request(
                music_media_id=TRACK,
                music_duration_ms=180_000,
                beats=grid(120.0, start=1_250),
                beat_sync=True,
            ),
            candidates(),
        )
        assert outcome.plan.music is not None
        assert outcome.plan.music.source_in_ms == 1_250

    def test_the_cue_starts_at_zero_without_a_grid(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(music_media_id=TRACK, music_duration_ms=180_000), candidates()
        )
        assert outcome.plan.music is not None
        assert outcome.plan.music.source_in_ms == 0

    def test_the_cue_is_trimmed_to_the_track(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(music_media_id=TRACK, music_duration_ms=9_000), candidates()
        )
        assert outcome.plan.music is not None
        assert outcome.plan.music.source_out_ms <= 9_000

    def test_a_track_too_short_to_use_is_dropped_rather_than_failing_the_plan(self) -> None:
        """A cue the validator would reject would fail the whole edit over the
        bed; dropping it produces a silent edit, which is recoverable."""
        outcome = RulesEnginePlanner().plan(
            request(music_media_id=TRACK, music_duration_ms=100), candidates()
        )
        assert outcome.plan.music is None

    def test_the_fade_out_is_shortened_to_fit_a_short_cue(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(
                music_media_id=TRACK,
                music_duration_ms=2_000,
                music_fade_in_ms=500,
                music_fade_out_ms=5_000,
            ),
            candidates(),
        )
        cue = outcome.plan.music
        assert cue is not None
        assert cue.fade_in_ms + cue.fade_out_ms <= cue.duration_ms

    def test_gain_and_fades_come_from_the_request(self) -> None:
        outcome = RulesEnginePlanner().plan(
            request(
                music_media_id=TRACK,
                music_duration_ms=180_000,
                music_gain=0.4,
                music_fade_in_ms=750,
                music_fade_out_ms=2_000,
            ),
            candidates(),
        )
        cue = outcome.plan.music
        assert cue is not None
        assert cue.gain == pytest.approx(0.4)
        assert cue.fade_in_ms == 750
        assert cue.fade_out_ms == 2_000


# --------------------------------------------------------------- determinism
class TestDeterminism:
    def test_two_runs_over_one_input_agree(self) -> None:
        """The property the whole rules engine exists to have, with beats on."""
        clips = candidates()
        first = RulesEnginePlanner().plan(request(beats=grid(128.0), beat_sync=True), clips)
        second = RulesEnginePlanner().plan(request(beats=grid(128.0), beat_sync=True), clips)
        assert first.plan.as_payload() == second.plan.as_payload()
