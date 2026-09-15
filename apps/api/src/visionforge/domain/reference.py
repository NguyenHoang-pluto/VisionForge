"""What a reference video's style actually measures as.

A reference video is a clip the user already owns, nominated as "make my edit
feel like this". Phase 8 reads it with the analyzers that already exist and
turns their rows into a small, versioned, entirely numeric description:

    scenes  + beats     -> pacing, cut rate, beat-sync tendency
    quality             -> luminance, contrast
    dynamics            -> motion energy, saturation

Three rules govern this module, and they are the whole design.

**Every number is measured.** There is no feature here that is inferred from
another, defaulted when missing, or filled in from the style the user picked. A
signal that was not analysed is absent, and absent is a value the rest of the
system handles -- it is not zero, and it is not the population mean.

**Every number carries its own confidence.** A cut rate computed from two
detected scenes is not the same claim as one computed from forty, and a palette
read from one decodable frame is not the same claim as one read from five. The
confidence travels with the measurement so the policy layer can weight by it
rather than treating every figure as equally trustworthy.

**Nothing here is a template.** The profile describes footage the user supplied.
It carries no frames, no filename, no storage key and no text -- only numbers --
which is what makes it safe to hand to a model and what stops "style transfer"
from meaning "use the reference's pictures".

Pure domain: no SQL, no OpenCV, no I/O. It consumes payload dictionaries that
the analyzers already wrote and the repository already read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from visionforge.domain.beats import MIN_BEAT_CONFIDENCE, BeatGrid
from visionforge.domain.ids import MediaId
from visionforge.domain.selection import Candidate

#: Bumped whenever the meaning of a field changes, so a stored or cached profile
#: from an older build is recognisable rather than silently misread. The profile
#: is derived on demand from analysis rows, so a bump costs a recomputation and
#: never a migration.
PROFILE_VERSION = "1"

#: Cuts needed before pacing is characterised at full confidence.
#:
#: One scene means the detector found nothing -- which the scene analyzer
#: reports as a single scene spanning the clip -- and that is indistinguishable
#: from a genuine single take. Eight cuts is where the median shot length stops
#: moving much when one more arrives.
PACING_FULL_CUTS = 8

#: Cuts needed before beat-sync tendency means anything at all. Two cuts can
#: coincide with a beat by chance often enough to be worthless.
MIN_CUTS_FOR_BEAT_TENDENCY = 3

#: How close a cut has to be to a beat to count as landing on it.
#:
#: Proportional to the beat period, because "on the beat" is a musical judgement
#: and a 90 ms error at 180 BPM is half a beat, but capped in absolute terms
#: because at very slow tempi a proportional window would accept almost
#: anything.
BEAT_WINDOW_RATIO = 0.18
MAX_BEAT_WINDOW_MS = 90


class Pacing(StrEnum):
    """A shot length, named. Derived from the median, never asserted."""

    SLOW = "slow"
    MEASURED = "measured"
    BRISK = "brisk"
    RAPID = "rapid"


#: Upper bound of each bucket, in milliseconds of median shot length.
PACING_BUCKETS: tuple[tuple[int, Pacing], ...] = (
    (900, Pacing.RAPID),
    (2_000, Pacing.BRISK),
    (4_500, Pacing.MEASURED),
)


def pacing_for(median_shot_ms: int) -> Pacing:
    for ceiling, pacing in PACING_BUCKETS:
        if median_shot_ms < ceiling:
            return pacing
    return Pacing.SLOW


@dataclass(frozen=True, slots=True)
class Measurement:
    """One number and how much it should be trusted.

    ``confidence`` is 0..1 and is about the *evidence*, not the value: a
    confidently-measured low energy is ``value=0.05, confidence=1.0``. The two
    are separate because the policy layer scales influence by the second and
    the direction of the effect by the first.
    """

    value: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def as_payload(self) -> dict[str, float]:
        return {"value": round(self.value, 4), "confidence": round(self.confidence, 3)}


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """The measured style of one reference video.

    Every field is ``None`` when the signal it needs was not analysed or was not
    interpretable. A consumer that wants a number must decide what to do without
    one; none of them may substitute a default, because a default is a claim
    about footage nobody looked at.
    """

    version: str
    media_id: MediaId
    source_duration_ms: int

    # --- cutting ---
    #: Median shot length. The single most characteristic number in an edit.
    shot_ms: Measurement | None = None
    #: Quartiles of shot length, in milliseconds. Shape, not just centre: an
    #: edit of uniform two-second shots and one alternating half-second and
    #: four-second shots have the same median and different styles.
    shot_ms_p25: int | None = None
    shot_ms_p75: int | None = None
    #: Cuts per minute.
    cut_rate: Measurement | None = None
    scene_count: int | None = None

    # --- look ---
    luminance: Measurement | None = None
    contrast: Measurement | None = None
    saturation: Measurement | None = None
    motion: Measurement | None = None

    # --- rhythm ---
    bpm: float | None = None
    beat_confidence: float | None = None
    #: Fraction of this reference's own cuts that land on its own beats. The
    #: measurement behind "this edit was cut to the music".
    beat_sync: Measurement | None = None

    @property
    def pacing(self) -> Pacing | None:
        return pacing_for(int(self.shot_ms.value)) if self.shot_ms else None

    @property
    def confidence(self) -> float:
        """How well characterised this reference is overall.

        The mean of what was actually measured, not of every field: a profile
        with four strong features and three absent ones is four-sevenths
        confident only if the absent ones were expected, and they are not. An
        empty profile is zero.
        """
        present = [
            m.confidence
            for m in (
                self.shot_ms,
                self.cut_rate,
                self.luminance,
                self.contrast,
                self.saturation,
                self.motion,
                self.beat_sync,
            )
            if m is not None
        ]
        return round(sum(present) / len(present), 3) if present else 0.0

    @property
    def is_usable(self) -> bool:
        """Whether this profile should be allowed to influence an edit at all.

        Pacing is the floor. Everything else refines an edit; without a shot
        length there is nothing to refine *towards*, and a profile that only
        knows the reference was bright would nudge colour weighting on the
        strength of one number.
        """
        return self.shot_ms is not None and self.shot_ms.confidence > 0.0

    def as_payload(self) -> dict[str, Any]:
        """The wire and prompt form. Numbers, enums and nulls -- nothing else.

        This is what crosses both the HTTP boundary and the model boundary, and
        it is deliberately the same shape for both: there is no field a path, a
        key, a filename or an instruction could travel in, so the profile cannot
        carry an injection however the reference was named.
        """
        payload: dict[str, Any] = {
            "version": self.version,
            "duration_ms": self.source_duration_ms,
            "confidence": self.confidence,
            "usable": self.is_usable,
        }
        for name, measurement in (
            ("shot_ms", self.shot_ms),
            ("cut_rate", self.cut_rate),
            ("luminance", self.luminance),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("motion", self.motion),
            ("beat_sync", self.beat_sync),
        ):
            payload[name] = measurement.as_payload() if measurement else None

        payload["pacing"] = self.pacing.value if self.pacing else None
        payload["scene_count"] = self.scene_count
        payload["shot_ms_p25"] = self.shot_ms_p25
        payload["shot_ms_p75"] = self.shot_ms_p75
        payload["bpm"] = self.bpm
        payload["beat_confidence"] = self.beat_confidence
        return payload


# --------------------------------------------------------------- construction
def _quantile(values: list[int], fraction: float) -> int:
    """Nearest-rank quantile.

    Written out rather than pulled from numpy because this module is domain
    code and may not import it, and because nearest-rank on a handful of shot
    lengths is both exact and obvious -- there is no interpolation to disagree
    about between one build and the next.
    """
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _pacing_confidence(cut_count: int) -> float:
    """How much a shot-length distribution of this size is worth."""
    if cut_count <= 0:
        return 0.0
    return round(min(1.0, cut_count / PACING_FULL_CUTS), 3)


def _cut_times(scenes: list[dict[str, Any]]) -> list[int]:
    """Where the cuts are: the start of every scene after the first.

    A scene's start is a cut; the clip's own start is not.
    """
    times: list[int] = []
    for scene in scenes[1:]:
        start = scene.get("start_ms")
        if isinstance(start, int | float):
            times.append(int(start))
    return sorted(times)


def beat_sync_tendency(cut_times_ms: list[int], grid: BeatGrid) -> Measurement | None:
    """What fraction of these cuts land on one of these beats.

    The measurement behind "this reference was cut to its music". Returns
    ``None`` -- not zero -- when there is not enough to judge: too few cuts, or
    a grid the domain does not trust. Zero would mean "measured, and it was not
    cut to the music", which is a different and much stronger claim.
    """
    if len(cut_times_ms) < MIN_CUTS_FOR_BEAT_TENDENCY:
        return None
    if not grid.beats_ms or not grid.is_reliable():
        return None

    period = grid.period_ms
    if period <= 0:
        return None
    window = min(MAX_BEAT_WINDOW_MS, period * BEAT_WINDOW_RATIO)

    beats = sorted(grid.beats_ms)
    on_beat = 0
    for cut in cut_times_ms:
        # Nearest beat by scan: a reference has at most a few thousand beats and
        # a few hundred cuts, so the clarity is worth more than the bisect.
        nearest = min(beats, key=lambda beat: abs(beat - cut))
        if abs(nearest - cut) <= window:
            on_beat += 1

    fraction = on_beat / len(cut_times_ms)
    # Confident in proportion to how many cuts were judged and how much the grid
    # itself is trusted: a strong tendency measured over four cuts against a
    # marginal grid is a weak claim, and says so.
    evidence = min(1.0, len(cut_times_ms) / PACING_FULL_CUTS)
    return Measurement(value=round(fraction, 4), confidence=round(evidence * grid.confidence, 3))


def _measurement(payload: dict[str, Any], key: str, confidence: float) -> Measurement | None:
    value = payload.get(key)
    if not isinstance(value, int | float) or confidence <= 0:
        return None
    return Measurement(value=float(value), confidence=round(min(1.0, confidence), 3))


def profile_from(
    *,
    media_id: MediaId,
    duration_ms: int,
    scenes: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    dynamics: dict[str, Any] | None = None,
    beats: dict[str, Any] | None = None,
) -> ReferenceProfile:
    """Build a profile from whichever analysis payloads exist.

    Deterministic: the same payloads always produce the same profile, and every
    missing payload subtracts features rather than changing the ones that
    remain. That property is what the determinism tests pin, and it is what lets
    the profile be recomputed on read instead of stored.
    """
    shot_ms: Measurement | None = None
    cut_rate: Measurement | None = None
    p25: int | None = None
    p75: int | None = None
    scene_count: int | None = None
    cut_times: list[int] = []

    # ------------------------------------------------------------- cutting
    if scenes:
        raw = scenes.get("scenes")
        entries = [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []
        durations = [
            int(s["duration_ms"])
            for s in entries
            if isinstance(s.get("duration_ms"), int | float) and s["duration_ms"] > 0
        ]
        scene_count = len(entries) or None
        cut_times = _cut_times(entries)

        # One scene means "the detector found no cuts", which the analyzer
        # reports as a single scene spanning the clip. That is a real
        # observation about a single take and a null observation about pacing,
        # and the confidence of zero is what says so.
        confidence = _pacing_confidence(len(durations) - 1)
        if durations and confidence > 0:
            shot_ms = Measurement(value=float(_quantile(durations, 0.5)), confidence=confidence)
            p25 = _quantile(durations, 0.25)
            p75 = _quantile(durations, 0.75)

            minutes = duration_ms / 60_000 if duration_ms > 0 else 0.0
            if minutes > 0:
                cut_rate = Measurement(
                    value=round(len(cut_times) / minutes, 3), confidence=confidence
                )

    # ---------------------------------------------------------------- look
    luminance = contrast = None
    if quality:
        frames = quality.get("frame_count")
        seen = float(frames) if isinstance(frames, int | float) else 0.0
        look_confidence = min(1.0, seen / 5.0)
        # Luminance arrives 0..255 from the analyzer and is normalised here, so
        # every field on the profile is on the same 0..1 scale and a consumer
        # never has to remember which one is not.
        raw_luma = quality.get("mean_luminance")
        if isinstance(raw_luma, int | float) and look_confidence > 0:
            luminance = Measurement(
                value=round(float(raw_luma) / 255.0, 4), confidence=round(look_confidence, 3)
            )
        raw_contrast = quality.get("contrast")
        if isinstance(raw_contrast, int | float) and look_confidence > 0:
            # Standard deviation of an 8-bit channel: 128 is the theoretical
            # maximum for a two-level image, and real footage never approaches
            # it, so the normalisation is a scale rather than a claim.
            contrast = Measurement(
                value=round(min(1.0, float(raw_contrast) / 128.0), 4),
                confidence=round(look_confidence, 3),
            )

    saturation = motion = None
    if dynamics:
        colour_confidence = dynamics.get("colour_confidence")
        motion_confidence = dynamics.get("motion_confidence")
        saturation = _measurement(
            dynamics,
            "saturation",
            float(colour_confidence) if isinstance(colour_confidence, int | float) else 0.0,
        )
        motion = _measurement(
            dynamics,
            "motion",
            float(motion_confidence) if isinstance(motion_confidence, int | float) else 0.0,
        )

    # -------------------------------------------------------------- rhythm
    bpm: float | None = None
    beat_confidence: float | None = None
    beat_sync: Measurement | None = None
    if beats:
        raw_bpm = beats.get("bpm")
        raw_conf = beats.get("confidence")
        if isinstance(raw_bpm, int | float) and isinstance(raw_conf, int | float):
            bpm = round(float(raw_bpm), 2)
            beat_confidence = round(float(raw_conf), 3)

            positions = beats.get("beats_ms")
            if isinstance(positions, list) and positions and raw_bpm > 0:
                grid = BeatGrid(
                    bpm=float(raw_bpm),
                    confidence=min(1.0, max(0.0, float(raw_conf))),
                    beats_ms=tuple(int(p) for p in positions if isinstance(p, int | float)),
                    source_duration_ms=duration_ms or None,
                )
                beat_sync = beat_sync_tendency(cut_times, grid)

    return ReferenceProfile(
        version=PROFILE_VERSION,
        media_id=media_id,
        source_duration_ms=duration_ms,
        shot_ms=shot_ms,
        shot_ms_p25=p25,
        shot_ms_p75=p75,
        cut_rate=cut_rate,
        scene_count=scene_count,
        luminance=luminance,
        contrast=contrast,
        saturation=saturation,
        motion=motion,
        bpm=bpm,
        beat_confidence=beat_confidence,
        beat_sync=beat_sync,
    )


def without_reference(
    candidates: list[Candidate], reference_media_id: MediaId | None
) -> list[Candidate]:
    """Drop the reference from the footage a plan may choose from.

    Style transfer here means "cut my clips the way that clip was cut". It has
    never meant "put that clip in my edit", and a reference that was silently
    eligible for selection would do exactly that -- most often winning, since a
    professionally-cut reference outscores a user's phone footage on every
    usability signal the ranker has.

    Applied where candidates are built, so it covers the rules engine and the
    model alike: neither can select what it was never handed, and the model is
    additionally never given a handle for it.
    """
    if reference_media_id is None:
        return candidates
    return [c for c in candidates if c.media_id != reference_media_id]


__all__ = [
    "MIN_BEAT_CONFIDENCE",
    "MIN_CUTS_FOR_BEAT_TENDENCY",
    "PACING_FULL_CUTS",
    "PROFILE_VERSION",
    "Measurement",
    "Pacing",
    "ReferenceProfile",
    "beat_sync_tendency",
    "pacing_for",
    "profile_from",
    "without_reference",
]
