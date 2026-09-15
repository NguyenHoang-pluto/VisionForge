"""Beat detection, measured against signals whose tempo is known by construction.

Synthesised rather than recorded: a click track generated at exactly 120 BPM has
a right answer, so "detected 119.4" is a measurable error rather than an opinion.
Real music has no ground truth available to a test, which is why the integration
suite checks a rendered file and this one checks the arithmetic.

The negative cases carry as much weight as the positive ones. Silence, noise and
a steady hum must all come back with low confidence, because the planner's
fallback is driven by that number and a detector that is confident about noise
is worse than one that finds nothing.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from visionforge.domain.analysis import AnalysisSource, AnalysisStatus
from visionforge.domain.beats import MIN_BEAT_CONFIDENCE, BeatGrid, grid_from_payload
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.beats import (
    HOP_MS,
    SAMPLE_RATE,
    BeatAnalyzer,
    detect_beats,
    estimate_tempo,
    onset_envelope,
)


# --------------------------------------------------------------------- signals
def click_track(bpm: float, seconds: float = 12.0, *, offset_ms: int = 0) -> np.ndarray:
    """A percussive click at a fixed tempo.

    Each click is a short burst of broadband noise with an exponential decay --
    broadband because spectral flux measures energy appearing across *many* bins
    at once, and a pure sine would excite one.
    """
    total = int(SAMPLE_RATE * seconds)
    signal = np.zeros(total, dtype=np.float32)

    rng = np.random.default_rng(1234)  # fixed: the test must not be flaky
    click_len = int(SAMPLE_RATE * 0.03)
    burst = rng.standard_normal(click_len).astype(np.float32)
    burst *= np.exp(-np.linspace(0, 12, click_len)).astype(np.float32)

    period = SAMPLE_RATE * 60.0 / bpm
    start = int(offset_ms * SAMPLE_RATE / 1000)
    position = float(start)
    while position + click_len < total:
        at = int(round(position))
        signal[at : at + click_len] += burst
        position += period
    return signal


def silence(seconds: float = 8.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def steady_tone(seconds: float = 8.0, hz: float = 220.0) -> np.ndarray:
    """A continuous tone: energy, but no onsets after the first."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def white_noise(seconds: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(99)
    return (0.2 * rng.standard_normal(int(SAMPLE_RATE * seconds))).astype(np.float32)


# ------------------------------------------------------------------- envelope
class TestOnsetEnvelope:
    def test_a_click_track_produces_peaks(self) -> None:
        envelope = onset_envelope(click_track(120.0))
        assert envelope.size > 0
        assert envelope.max() == pytest.approx(1.0)

    def test_silence_produces_a_flat_envelope(self) -> None:
        assert onset_envelope(silence()).max() == 0.0

    def test_a_steady_tone_produces_no_ongoing_onsets(self) -> None:
        """Energy is not an onset. Only energy *arriving* is.

        The first frames are excluded, and honestly so: half a window is
        prepended to centre frame *i* on sample *i*hop*, and reflecting there
        leaves a fold -- a corner, not a step, but still broadband. It shows up
        as one artefact at the very start of any signal. It is one sample among
        hundreds, it cannot make a grid periodic, and the track-level tests
        below show a tone is still rejected. What matters is that *after* the
        boundary a sustained tone contributes essentially nothing.
        """
        envelope = onset_envelope(steady_tone())
        assert envelope[4:].max() < 0.2
        assert envelope[4:].mean() < 0.02

    def test_too_short_a_signal_yields_nothing_rather_than_raising(self) -> None:
        assert onset_envelope(np.zeros(64, dtype=np.float32)).size == 0

    def test_the_envelope_is_never_negative(self) -> None:
        """Half-wave rectification: a decay must not read as an onset."""
        assert onset_envelope(click_track(90.0)).min() >= 0.0


# ---------------------------------------------------------------------- tempo
class TestTempoEstimation:
    @pytest.mark.parametrize("bpm", [90.0, 100.0, 120.0, 140.0, 160.0])
    def test_recovers_the_tempo_of_a_click_track(self, bpm: float) -> None:
        """Within 1 BPM.

        An integer lag alone would only resolve tempo to steps of several BPM at
        this hop; the parabolic fit around the autocorrelation peak is what
        makes this tight, and a regression in it shows up here first.
        """
        detected, _ = estimate_tempo(onset_envelope(click_track(bpm)))
        assert detected == pytest.approx(bpm, abs=1.0)

    @pytest.mark.parametrize("bpm", [120.0, 140.0])
    def test_does_not_halve_a_tempo_whose_period_straddles_a_frame(self, bpm: float) -> None:
        """The octave trap, pinned.

        At 120 BPM the period is 21.5 frames, so lag 43 aligns exactly while the
        fundamental splits across lags 21 and 22 -- and plain argmax picks 60
        BPM. The tempo prior is what breaks that tie, and this is the test that
        fails if it is removed.
        """
        detected, _ = estimate_tempo(onset_envelope(click_track(bpm)))
        assert detected > bpm * 0.75

    def test_reports_no_tempo_for_silence(self) -> None:
        bpm, confidence = estimate_tempo(onset_envelope(silence()))
        assert bpm == 0.0
        assert confidence == 0.0

    def test_reports_no_tempo_for_too_short_an_envelope(self) -> None:
        assert estimate_tempo(np.zeros(4)) == (0.0, 0.0)


# ------------------------------------------------------------------- detection
class TestDetectBeats:
    def test_a_click_track_is_detected_confidently(self) -> None:
        bpm, confidence, beats = detect_beats(click_track(120.0, seconds=16.0))

        assert bpm == pytest.approx(120.0, abs=2.0)
        assert confidence >= MIN_BEAT_CONFIDENCE
        # 16 s at 120 BPM is 32 beats; allow the ends to be clipped.
        assert 28 <= len(beats) <= 34

    def test_the_grid_lands_on_the_clicks(self) -> None:
        """The property that matters: not just the right tempo, the right phase."""
        _, _, beats = detect_beats(click_track(120.0, seconds=16.0, offset_ms=250))

        period = 500.0
        for beat in beats[:8]:
            # Within one hop (~23 ms) of a real click at 250 + 500k. The hop is
            # the floor on precision: the envelope has no finer resolution.
            offset = (beat - 250) % period
            assert min(offset, period - offset) <= HOP_MS + 2

    def test_beats_are_ascending_and_unique(self) -> None:
        _, _, beats = detect_beats(click_track(128.0))
        assert beats == sorted(set(beats))

    def test_silence_is_not_confidently_anything(self) -> None:
        _, confidence, _ = detect_beats(silence())
        assert confidence < MIN_BEAT_CONFIDENCE

    def test_a_steady_tone_is_not_confidently_anything(self) -> None:
        """An engine hum is periodic. It is not a beat, and the second opinion
        (does the grid sit on onsets?) is what tells them apart."""
        _, confidence, _ = detect_beats(steady_tone())
        assert confidence < MIN_BEAT_CONFIDENCE

    def test_white_noise_is_not_confidently_anything(self) -> None:
        _, confidence, _ = detect_beats(white_noise())
        assert confidence < MIN_BEAT_CONFIDENCE

    def test_detection_is_deterministic(self) -> None:
        """The whole reason this is arithmetic and not a model."""
        signal = click_track(110.0)
        assert detect_beats(signal) == detect_beats(signal)

    def test_an_empty_signal_returns_nothing_rather_than_raising(self) -> None:
        assert detect_beats(np.zeros(0, dtype=np.float32)) == (0.0, 0.0, [])


# ------------------------------------------------------- feeding the domain
class TestFeedsTheDomain:
    def test_a_detected_grid_is_usable_by_the_planner(self) -> None:
        bpm, confidence, beats = detect_beats(click_track(120.0, seconds=16.0))
        grid = BeatGrid(bpm=bpm, confidence=confidence, beats_ms=tuple(beats))

        assert grid.is_reliable()
        # A 2.5 s target at 120 BPM is exactly 5 beats.
        assert grid.snap_span_ms(2_400, min_ms=300, max_ms=30_000) == pytest.approx(2_500, abs=40)

    def test_a_detection_on_noise_is_refused_by_the_domain(self) -> None:
        bpm, confidence, beats = detect_beats(white_noise())
        grid = BeatGrid(bpm=max(bpm, 0.0), confidence=confidence, beats_ms=tuple(beats))
        assert not grid.is_reliable()

    def test_the_payload_round_trips_into_a_grid(self) -> None:
        bpm, confidence, beats = detect_beats(click_track(100.0))
        payload = {
            "bpm": bpm,
            "confidence": confidence,
            "beats_ms": beats,
            "source_duration_ms": 12_000,
        }
        restored = grid_from_payload(payload)
        assert restored is not None
        assert restored.beat_count == len(beats)


# ------------------------------------------------------------------- analyzer
class TestAnalyzerContract:
    def test_it_analyses_anything_with_a_soundtrack(self) -> None:
        """Audio and video, not images.

        Phase 7 restricted this to audio because a music bed was the only thing
        a tempo was wanted for. Phase 8 reads a reference video's cutting
        rhythm, which is only interpretable against the music that video was cut
        to -- so the grid has to exist for the video itself. FFmpeg pulls the
        audio stream out of the container either way.
        """
        analyzer = BeatAnalyzer()
        assert analyzer.supports(MediaKind.AUDIO)
        assert analyzer.supports(MediaKind.VIDEO)
        assert not analyzer.supports(MediaKind.IMAGE)

    def test_images_are_unsupported_rather_than_failed(self) -> None:
        """A still having no tempo is a fact about the medium, not an error --
        and an 'unsupported' row is what stops it being retried forever."""
        outcome = BeatAnalyzer().analyze(
            AnalysisSource(
                media_id=MediaId(uuid.uuid4()),
                kind=MediaKind.IMAGE,
                local_path="/does/not/matter.png",
                used_proxy=False,
            )
        )
        assert outcome.status is AnalysisStatus.UNSUPPORTED
        assert "reason" in outcome.payload

    def test_the_version_is_pinned(self) -> None:
        """A retuning has to be a new row, not a silent edit to an old one."""
        assert BeatAnalyzer().version == "1"
