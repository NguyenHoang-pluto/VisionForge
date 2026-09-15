"""Deterministic beat detection.

No model, no network, no music API -- the audio is decoded locally and the tempo
is measured from it with arithmetic. Same file in, same beats out, forever,
which is what makes a beat-synced edit reproducible and a change in the cut
attributable to a parameter rather than to chance.

The method is the standard onset-envelope pipeline, in four steps:

1. **Decode** to mono PCM at a low rate through FFmpeg. 22050 Hz is well above
   what a tempo estimate needs and a quarter of the samples of CD audio.
2. **Onset envelope.** A short-time Fourier magnitude spectrogram, then spectral
   flux: the sum of positive frame-to-frame changes in each bin. Energy rising
   across many bins at once is what a drum hit is; half-wave rectifying is what
   stops an energy *fall* from registering as one.
3. **Tempo** by autocorrelating that envelope. The lag with the strongest
   correlation inside the searched BPM range is the beat period.
4. **Phase** by trying every grid offset within one period and keeping the one
   whose beat positions land on the most onset energy.

What this is good at: steady electronic and pop music with a clear pulse, which
is what gets laid under a highlight reel. What it is **not** good at is set out
in the module's limitations note and in the ADR -- it does not follow tempo
changes, it cannot tell a downbeat from any other beat, and it reports the
confidence that lets a caller ignore it.

NumPy only. No librosa, no scipy: this is roughly eighty lines of array
arithmetic, and the alternative is a dependency tree larger than the rest of the
application for one function.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile

import numpy as np
from numpy.typing import NDArray

from visionforge.domain.analysis import (
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
)
from visionforge.domain.beats import MAX_BEATS, MAX_BPM, MIN_BPM
from visionforge.domain.errors import UnsupportedMediaError
from visionforge.domain.media import MediaKind
from visionforge.infra.ffmpeg.runner import FFMPEG, resolve_binary

logger = logging.getLogger(__name__)

#: Bumped whenever the algorithm or a parameter below changes, so that an old
#: result stays interpretable and a retuning is a new row rather than a silent
#: edit to an existing one.
BEATS_ANALYZER_VERSION = "1"

# ------------------------------------------------------------------ parameters
#: Analysis sample rate. High enough that percussive transients survive, low
#: enough that a five-minute track is 6.6 M samples rather than 26 M.
SAMPLE_RATE = 22_050
#: FFT window. 2048 samples at 22050 Hz is ~93 ms -- long enough for usable
#: frequency resolution, short enough to localise a transient.
FRAME_SIZE = 2048
#: 512 samples is ~23 ms between envelope points, which bounds how precisely any
#: beat can be placed. Finer costs time for accuracy the cut does not use.
HOP_SIZE = 512
#: Envelope smoothing, in frames. Removes single-frame spikes from clicks and
#: vinyl noise without blurring a real onset.
SMOOTHING_FRAMES = 3
#: Decoding is bounded: a corrupt or pathological file must fail rather than
#: occupy a worker.
DECODE_TIMEOUT_S = 300.0
#: Longest span analysed. Tempo is measured from the opening minutes; a track
#: longer than this has its grid extrapolated, which is both cheaper and no less
#: accurate for music that holds a steady tempo (and no more wrong for music
#: that does not, since the estimate was already wrong).
MAX_ANALYSIS_MS = 300_000

HOP_MS = HOP_SIZE * 1000.0 / SAMPLE_RATE

#: Where the tempo prior is centred, and how wide it is in octaves.
#:
#: Autocorrelation cannot tell a tempo from half of it: the peak at twice the
#: true period is real, and for a period that is not a whole number of frames it
#: is often the *stronger* of the two, because the doubled lag lands on an
#: integer while the fundamental straddles one. Something has to break the tie,
#: and "music is usually nearer 120 BPM than 60" is the assumption every tempo
#: estimator makes. It is a bias, stated here rather than hidden: a genuine
#: 60 BPM track needs its fundamental to out-correlate the 120 BPM lag by about
#: a third, which a real half-time groove does.
TEMPO_PRIOR_CENTRE_BPM = 120.0
TEMPO_PRIOR_OCTAVES = 0.7


# ---------------------------------------------------------------------- decode
def decode_pcm(path: str, *, max_ms: int = MAX_ANALYSIS_MS) -> NDArray[np.float32]:
    """Decode any audio FFmpeg understands to mono float32 at ``SAMPLE_RATE``.

    Written to a temp file rather than read from a pipe: a five-minute track is
    26 MB of float32, and holding that in a pipe buffer while FFmpeg blocks on a
    full pipe is the classic way to deadlock a subprocess. The file is removed
    in every path.

    argv, never a shell string -- the same posture as every other FFmpeg call in
    this codebase, and it matters as much here because the path comes from a
    storage key.
    """
    handle, raw_path = tempfile.mkstemp(prefix="vf-beats-", suffix=".f32")
    os.close(handle)
    try:
        subprocess.run(
            [
                resolve_binary(FFMPEG),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                path,
                "-map",
                "a:0",
                "-t",
                f"{max_ms / 1000:.3f}",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                raw_path,
            ],
            check=True,
            capture_output=True,
            timeout=DECODE_TIMEOUT_S,
        )
        samples = np.fromfile(raw_path, dtype="<f4")
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-4:]
        raise UnsupportedMediaError(
            "could not decode an audio stream from this file",
            hint="\n".join(tail) or None,
        ) from exc
    finally:
        # The file is ours and exists; suppressing is for the case where the
        # decode was killed and something else already cleaned up.
        with contextlib.suppress(OSError):
            os.unlink(raw_path)

    # Non-finite samples appear in damaged files and would poison every mean
    # and correlation downstream.
    return np.nan_to_num(samples.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------------------------------------------- onset envelope
def onset_envelope(samples: NDArray[np.float32]) -> NDArray[np.float64]:
    """Spectral flux: how much energy is *rising* at each moment.

    Half-wave rectified on purpose. A drum hit is energy appearing across many
    bins at once; the decay afterwards is energy disappearing, and counting that
    too would put a second, phantom onset after every real one.

    Normalised to a 0..1 peak at the end so that the correlation and peak-picking
    thresholds below are independent of how loud the track was mastered.
    """
    if samples.size < FRAME_SIZE * 2:
        return np.zeros(0, dtype=np.float64)

    window = np.hanning(FRAME_SIZE).astype(np.float64)

    # Pad the front by half a window so that frame *i* is centred on sample
    # ``i * HOP_SIZE`` rather than starting there. Without this every detected
    # beat lands early by half a window -- about 46 ms, which is two frames and
    # plainly audible against a click track. Centring is what makes
    # ``_frame_to_ms`` below a one-liner instead of an empirical fudge.
    # Reflected, not zero-filled. A zero pad puts a step discontinuity at the
    # very start of every signal, which reads as the loudest onset in the track
    # and then sets the scale everything else is normalised against. Reflecting
    # continues the waveform instead, so the padding contributes no flux.
    padded = np.pad(samples, (FRAME_SIZE // 2, 0), mode="reflect")
    frame_count = 1 + (padded.size - FRAME_SIZE) // HOP_SIZE

    # One strided view over the signal rather than a Python loop building
    # frames: this is the only part of the pipeline whose cost scales with
    # track length, and a copy per frame would dominate it.
    strided = np.lib.stride_tricks.sliding_window_view(padded, FRAME_SIZE)[::HOP_SIZE]
    frames: NDArray[np.float64] = strided[:frame_count].astype(np.float64) * window

    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    # Log compression: without it a loud chorus contributes far more flux than a
    # quiet verse and the tempo estimate follows the arrangement, not the beat.
    magnitude = np.log1p(magnitude)

    flux = np.diff(magnitude, axis=0)
    envelope = np.sum(np.maximum(flux, 0.0), axis=1)

    if SMOOTHING_FRAMES > 1 and envelope.size >= SMOOTHING_FRAMES:
        kernel = np.ones(SMOOTHING_FRAMES) / SMOOTHING_FRAMES
        envelope = np.convolve(envelope, kernel, mode="same")

    envelope -= envelope.mean()
    envelope = np.maximum(envelope, 0.0)
    peak = float(envelope.max()) if envelope.size else 0.0
    normalised: NDArray[np.float64] = envelope / peak if peak > 0 else envelope
    return normalised


# ------------------------------------------------------------------ tempo
def estimate_tempo(envelope: NDArray[np.float64]) -> tuple[float, float]:
    """Beats per minute, and how strongly the envelope agrees.

    Autocorrelation over the lags corresponding to ``MIN_BPM``..``MAX_BPM``. The
    lag with the strongest correlation is the beat period; the correlation
    itself, normalised against zero lag, is the agreement.

    Returns ``(0.0, 0.0)`` when there is not enough signal to measure, which the
    caller reports as an unusable grid rather than guessing.
    """
    if envelope.size < 16:
        return 0.0, 0.0

    min_lag = max(1, int(round(60_000.0 / MAX_BPM / HOP_MS)))
    max_lag = int(round(60_000.0 / MIN_BPM / HOP_MS))
    max_lag = min(max_lag, envelope.size - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0

    centred = envelope - envelope.mean()
    full = np.correlate(centred, centred, mode="full")
    autocorr = full[full.size // 2 :]
    zero = float(autocorr[0])
    if zero <= 0:
        return 0.0, 0.0

    lags = np.arange(min_lag, max_lag + 1)
    if lags.size == 0:
        return 0.0, 0.0

    correlation = autocorr[lags] / zero
    candidate_bpm = 60_000.0 / (lags * HOP_MS)
    prior = np.exp(
        -0.5 * (np.log2(candidate_bpm / TEMPO_PRIOR_CENTRE_BPM) / TEMPO_PRIOR_OCTAVES) ** 2
    )

    best = int(np.argmax(correlation * prior))
    lag = float(lags[best])

    # Sub-frame refinement. The hop is ~23 ms, so an integer lag quantises the
    # tempo to steps of several BPM; fitting a parabola through the winning lag
    # and its neighbours recovers the fractional period the peak is really at,
    # which is what takes a 120 BPM track from "about 117" to "120.0".
    index = best + min_lag
    if 0 < index < autocorr.size - 1:
        left, centre, right = (
            float(autocorr[index - 1]),
            float(autocorr[index]),
            float(autocorr[index + 1]),
        )
        denominator = left - 2.0 * centre + right
        if denominator != 0:
            shift = 0.5 * (left - right) / denominator
            # Only trust the fit inside one sample, which is where a parabola
            # through three points is a reasonable model of the peak.
            if -1.0 < shift < 1.0:
                lag += shift

    if lag <= 0:
        return 0.0, 0.0

    strength = float(correlation[best])
    bpm = 60_000.0 / (lag * HOP_MS)
    return bpm, max(0.0, min(1.0, strength))


def _frame_to_ms(frame: int) -> float:
    """When, in the source, an envelope sample happened.

    ``+1`` because the envelope is a *difference* between consecutive
    spectrogram frames: flux index *e* is the energy that arrived between frame
    *e* and frame *e+1*, so it belongs to the later one. With the front padding
    in ``onset_envelope``, frame *i* is centred on sample ``i * HOP_SIZE``, and
    the whole conversion is this.
    """
    return (frame + 1) * HOP_MS


def _phase_offset(envelope: NDArray[np.float64], period_frames: float) -> int:
    """The grid offset, in frames, that lands on the most onset energy.

    Every candidate within one period is tried and scored by the total envelope
    value at its beat positions. Exhaustive rather than clever: one period is at
    most ~43 frames at 60 BPM, so the search is trivially cheap and has no
    tie-breaking subtleties to get wrong. Ties resolve to the earliest offset.
    """
    if period_frames <= 0 or envelope.size == 0:
        return 0

    best_offset, best_score = 0, -1.0
    for offset in range(max(1, int(round(period_frames)))):
        frames = _grid_frames(envelope.size, offset, period_frames)
        if frames.size == 0:
            continue
        score = float(envelope[frames].sum())
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset


def _grid_frames(size: int, offset: int, period_frames: float) -> NDArray[np.intp]:
    """Frame indices of a grid, guaranteed to be inside the envelope.

    ``arange`` with a float step then rounding can land exactly on ``size``,
    which is one past the end. Clipping after rounding rather than bounding the
    arange keeps the two paths (scoring a phase, emitting beats) identical.
    """
    if period_frames <= 0 or size <= 0:
        return np.zeros(0, dtype=np.intp)
    positions = np.arange(offset, size, period_frames)
    frames = np.round(positions).astype(np.intp)
    return frames[(frames >= 0) & (frames < size)]


def _grid_agreement(envelope: NDArray[np.float64], beats_frames: NDArray[np.intp]) -> float:
    """How much of the envelope's energy the grid actually sits on.

    The tempo autocorrelation can be strong for audio with a regular *texture*
    and no beat at all -- a hum, an engine. This is the second opinion: the mean
    envelope value under the grid, against the mean everywhere. A grid that
    genuinely marks onsets scores well above 1; one that marks nothing in
    particular scores around 1.
    """
    if beats_frames.size == 0 or envelope.size == 0:
        return 0.0
    overall = float(envelope.mean())
    if overall <= 0:
        return 0.0
    on_beat = float(envelope[beats_frames].mean())
    # Mapped to 0..1 with 3x the mean counting as full agreement, which is about
    # where a clean drum track lands.
    return max(0.0, min(1.0, (on_beat / overall - 1.0) / 2.0))


def detect_beats(samples: NDArray[np.float32]) -> tuple[float, float, list[int]]:
    """Tempo, confidence and beat positions in milliseconds.

    The confidence is the product of two independent measures: how periodic the
    envelope is, and how well the resulting grid coincides with its peaks.
    """
    envelope = onset_envelope(samples)
    bpm, periodicity = estimate_tempo(envelope)
    if bpm <= 0:
        return 0.0, 0.0, []

    period_frames = 60_000.0 / bpm / HOP_MS
    offset = _phase_offset(envelope, period_frames)
    frames = _grid_frames(envelope.size, offset, period_frames)[:MAX_BEATS]

    # Two independent opinions, multiplied because both have to hold.
    # Periodicity alone calls an idling engine 128 BPM -- its leakage really is
    # periodic. Coincidence alone is satisfied by any grid fine enough to hit
    # everything. A signal has to pass both to move a cut.
    confidence = periodicity * _grid_agreement(envelope, frames)
    beats_ms = [int(round(_frame_to_ms(int(frame)))) for frame in frames]
    return bpm, max(0.0, min(1.0, confidence)), beats_ms


# ---------------------------------------------------------------- the analyzer
class BeatAnalyzer:
    """Tempo and a beat grid for a project's audio.

    Audio only. A video's soundtrack is deliberately out of scope: Phase 7 beds
    are project-owned audio assets, and detecting a tempo in dialogue would
    produce a confident number with no musical meaning.
    """

    name = AnalyzerName.BEATS
    version = BEATS_ANALYZER_VERSION
    kind = AnalyzerKind.CPU

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind is MediaKind.AUDIO

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"beat detection does not apply to {source.kind.value}"},
            )

        samples = decode_pcm(source.local_path)
        if samples.size == 0:
            raise UnsupportedMediaError("audio stream decoded to no samples")

        bpm, confidence, beats_ms = detect_beats(samples)
        analysed_ms = int(samples.size * 1000 / SAMPLE_RATE)

        logger.info(
            "beats analysed",
            extra={
                "media_id": str(source.media_id),
                "bpm": round(bpm, 2),
                "confidence": round(confidence, 3),
                "beats": len(beats_ms),
            },
        )

        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            # A track with no detectable pulse is a fact about the audio, not a
            # failure: the row is stored with zero confidence and the planner
            # declines to use it, which is the outcome that keeps a spoken-word
            # bed from silently moving every cut.
            status=AnalysisStatus.OK,
            payload={
                "bpm": round(bpm, 3),
                "confidence": round(confidence, 4),
                "beat_count": len(beats_ms),
                "beats_ms": beats_ms,
                "source_duration_ms": source.duration_ms,
                "analysed_ms": analysed_ms,
                "truncated": analysed_ms >= MAX_ANALYSIS_MS,
                # The parameters the numbers above depend on, stored with them.
                # A result whose tuning cannot be recovered cannot be compared
                # against a later one.
                "sample_rate": SAMPLE_RATE,
                "frame_size": FRAME_SIZE,
                "hop_size": HOP_SIZE,
                "hop_ms": round(HOP_MS, 4),
                "search_bpm": [MIN_BPM, MAX_BPM],
                "tempo_prior_bpm": TEMPO_PRIOR_CENTRE_BPM,
                "tempo_prior_octaves": TEMPO_PRIOR_OCTAVES,
                "method": "spectral-flux-autocorrelation",
            },
            metrics={"samples": int(samples.size)},
        )


__all__ = [
    "BEATS_ANALYZER_VERSION",
    "HOP_MS",
    "MAX_ANALYSIS_MS",
    "SAMPLE_RATE",
    "TEMPO_PRIOR_CENTRE_BPM",
    "BeatAnalyzer",
    "decode_pcm",
    "detect_beats",
    "estimate_tempo",
    "onset_envelope",
]
