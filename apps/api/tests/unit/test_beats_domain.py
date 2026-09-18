"""Beat-grid arithmetic.

Pure functions over a grid, so every case here is exact rather than
approximate. The interesting behaviour is not "does it snap" -- it is *when it
refuses to*: an unreliable grid, a span the bounds cannot fit, and a snap that
would move the cut further than the caller could have wanted must all fall back
to the unquantised length rather than produce a worse edit confidently.
"""

from __future__ import annotations

import pytest

from visionforge.domain.beats import (
    MAX_BPM,
    MIN_BEAT_CONFIDENCE,
    MIN_BPM,
    BeatGrid,
    grid_from_payload,
)


def grid(bpm: float = 120.0, confidence: float = 0.9, count: int = 32, start: int = 0) -> BeatGrid:
    period = 60_000.0 / bpm
    return BeatGrid(
        bpm=bpm,
        confidence=confidence,
        beats_ms=tuple(round(start + i * period) for i in range(count)),
        source_duration_ms=round(start + count * period),
    )


# ------------------------------------------------------------------ structure
class TestConstruction:
    def test_period_is_the_inverse_of_tempo(self) -> None:
        assert grid(bpm=120.0).period_ms == pytest.approx(500.0)
        assert grid(bpm=90.0).period_ms == pytest.approx(666.667, abs=0.01)

    def test_zero_tempo_has_no_period(self) -> None:
        assert BeatGrid(bpm=0.0, confidence=0.0, beats_ms=()).period_ms == 0.0

    def test_first_beat_is_not_assumed_to_be_zero(self) -> None:
        """A track with a pickup bar starts its grid late, and must say so."""
        assert grid(start=1234).first_beat_ms == 1234

    def test_rejects_unsorted_beats(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            BeatGrid(bpm=120.0, confidence=0.9, beats_ms=(1000, 500))

    def test_rejects_duplicate_beats(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            BeatGrid(bpm=120.0, confidence=0.9, beats_ms=(500, 500))

    def test_rejects_negative_beats(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            BeatGrid(bpm=120.0, confidence=0.9, beats_ms=(-1, 500))

    def test_rejects_confidence_outside_zero_to_one(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            BeatGrid(bpm=120.0, confidence=1.5, beats_ms=(0,))


# ----------------------------------------------------------------- reliability
class TestReliability:
    def test_a_steady_confident_grid_is_reliable(self) -> None:
        assert grid(confidence=0.8).is_reliable()

    def test_low_confidence_is_not(self) -> None:
        assert not grid(confidence=MIN_BEAT_CONFIDENCE - 0.01).is_reliable()

    def test_tempo_below_the_searched_range_is_not(self) -> None:
        """A tempo outside the range means detection clamped, not measured."""
        assert not grid(bpm=MIN_BPM - 1).is_reliable()

    def test_tempo_above_the_searched_range_is_not(self) -> None:
        assert not grid(bpm=MAX_BPM + 1).is_reliable()

    def test_a_single_beat_establishes_no_period(self) -> None:
        assert not BeatGrid(bpm=120.0, confidence=1.0, beats_ms=(0,)).is_reliable()


# --------------------------------------------------------------------- queries
class TestQueries:
    def test_nearest_beat_picks_the_closest(self) -> None:
        g = grid(bpm=120.0)  # beats every 500 ms
        assert g.nearest_beat_ms(1100) == 1000
        assert g.nearest_beat_ms(1400) == 1500

    def test_nearest_beat_resolves_ties_early(self) -> None:
        """Arbitrary but fixed: two runs over one input must agree."""
        assert grid(bpm=120.0).nearest_beat_ms(1250) == 1000

    def test_nearest_beat_on_an_empty_grid_is_the_position_itself(self) -> None:
        empty = BeatGrid(bpm=0.0, confidence=0.0, beats_ms=())
        assert empty.nearest_beat_ms(777) == 777

    def test_beats_between_is_half_open(self) -> None:
        g = grid(bpm=120.0)
        assert g.beats_between(1000, 2000) == (1000, 1500)


# -------------------------------------------------------------------- snapping
class TestSnapping:
    def test_snaps_to_a_whole_number_of_beats(self) -> None:
        # 120 BPM = 500 ms/beat; 2400 ms is nearest 5 beats = 2500 ms.
        assert grid(bpm=120.0).snap_span_ms(2400, min_ms=300, max_ms=30_000) == 2500

    def test_an_exact_multiple_is_unchanged(self) -> None:
        assert grid(bpm=120.0).snap_span_ms(2500, min_ms=300, max_ms=30_000) == 2500

    def test_butt_joined_clips_of_equal_span_keep_every_cut_on_a_beat(self) -> None:
        """The whole argument for quantising duration rather than position.

        Clips abut, so if each is a whole number of beats long then every
        boundary is a beat boundary -- with no per-cut search and no drift.
        """
        g = grid(bpm=120.0, count=64)
        span = g.snap_span_ms(3300, min_ms=300, max_ms=30_000)
        period = g.period_ms
        for index in range(8):
            boundary = span * index
            assert boundary % period == pytest.approx(0.0, abs=0.5)

    def test_falls_back_when_the_grid_is_unreliable(self) -> None:
        unreliable = grid(confidence=0.1)
        assert unreliable.snap_span_ms(2400, min_ms=300, max_ms=30_000) == 2400

    def test_clamps_rather_than_snapping_outside_the_bounds(self) -> None:
        """Renderer bounds win over musical ones; they are not negotiable."""
        g = grid(bpm=120.0)
        assert g.snap_span_ms(60_000, min_ms=300, max_ms=4_000) == 4_000

    def test_refuses_a_snap_that_drifts_too_far(self) -> None:
        """A 60 BPM grid cannot honour a 700 ms clip without doubling it."""
        g = grid(bpm=60.0)  # 1000 ms/beat
        assert g.snap_span_ms(400, min_ms=300, max_ms=30_000) == 400

    def test_never_returns_zero_beats(self) -> None:
        g = grid(bpm=200.0)  # 300 ms/beat
        assert g.snap_span_ms(10, min_ms=300, max_ms=30_000) >= 300

    def test_is_deterministic(self) -> None:
        g = grid(bpm=137.0)
        first = [g.snap_span_ms(t, min_ms=300, max_ms=30_000) for t in range(1000, 9000, 137)]
        second = [g.snap_span_ms(t, min_ms=300, max_ms=30_000) for t in range(1000, 9000, 137)]
        assert first == second


# ----------------------------------------------------------------- persistence
class TestPayloadRoundTrip:
    def test_round_trips(self) -> None:
        original = grid(bpm=128.0, count=8, start=250)
        restored = grid_from_payload(original.as_payload())
        assert restored is not None
        assert restored.beats_ms == original.beats_ms
        assert restored.bpm == pytest.approx(original.bpm)

    def test_absent_payload_is_no_grid(self) -> None:
        assert grid_from_payload(None) is None
        assert grid_from_payload({}) is None

    def test_a_malformed_payload_is_no_grid_rather_than_an_error(self) -> None:
        """A bad analysis must degrade planning, not break it."""
        assert grid_from_payload({"bpm": "fast", "beats_ms": [0]}) is None

    def test_repairs_unsorted_and_duplicated_stored_beats(self) -> None:
        restored = grid_from_payload({"bpm": 120.0, "confidence": 0.9, "beats_ms": [500, 0, 500]})
        assert restored is not None
        assert restored.beats_ms == (0, 500)
