"""Beat grids, and the arithmetic of cutting to one.

Pure. No audio, no numpy, no file I/O -- the *detection* of beats is an
infrastructure concern (``infra.analysis.beats``) and lives behind the
``Analyzer`` port like every other signal. What lives here is the small amount
of arithmetic that decides where a cut lands once somebody has told us where the
beats are, because that arithmetic is a planning rule and has to be testable
without an audio file.

The distinction matters for one specific reason: a beat grid can be *wrong*.
Detection on a spoken-word recording, or on music with rubato, produces a tempo
that is confidently nonsense. Everything here therefore carries a confidence and
degrades to "do what Phase 4 did" rather than to a worse edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Tempo range searched by detection and accepted here. Below 60 BPM a "beat"
#: is longer than most clips in a highlight cut; above 200 the grid is finer
#: than the minimum segment length and quantising to it stops meaning anything.
MIN_BPM = 60.0
MAX_BPM = 200.0

#: Below this, a grid is treated as absent. Detection reports its own agreement
#: with the audio (see ``infra.analysis.beats``); this is the line at which we
#: stop letting it move a cut. Chosen so that a steady drum track passes and a
#: voice memo does not.
MIN_BEAT_CONFIDENCE = 0.35

#: How far from the target a snapped span may land before it is abandoned.
#: A cut moved by more than this is no longer the edit the caller asked for.
MAX_SNAP_DRIFT_RATIO = 0.5

#: Ceiling on how many beats are stored or returned. A ten-minute track at
#: 200 BPM is 2000 beats; the cap is a bound on payload size, not a musical
#: limit, and is far above anything a plan uses.
MAX_BEATS = 4000


@dataclass(frozen=True, slots=True)
class BeatGrid:
    """Where the beats are in one audio asset.

    ``beats_ms`` are absolute positions within the *source* file, ascending and
    unique. They are not assumed to start at zero: a track with a pickup bar has
    its first beat some way in, and cutting to a grid that pretends otherwise
    puts every cut consistently early.
    """

    bpm: float
    confidence: float
    beats_ms: tuple[int, ...]
    source_duration_ms: int | None = None

    def __post_init__(self) -> None:
        if self.bpm < 0:
            raise ValueError("bpm cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if any(b < 0 for b in self.beats_ms):
            raise ValueError("beat positions cannot be negative")
        if list(self.beats_ms) != sorted(set(self.beats_ms)):
            raise ValueError("beat positions must be ascending and unique")

    # ----------------------------------------------------------- properties
    @property
    def period_ms(self) -> float:
        """Milliseconds per beat. Zero when the tempo is unusable."""
        return 60_000.0 / self.bpm if self.bpm > 0 else 0.0

    @property
    def first_beat_ms(self) -> int:
        """Where the grid starts. ``0`` when there are no beats."""
        return self.beats_ms[0] if self.beats_ms else 0

    @property
    def beat_count(self) -> int:
        return len(self.beats_ms)

    def is_reliable(self, *, min_confidence: float = MIN_BEAT_CONFIDENCE) -> bool:
        """Whether this grid is trusted enough to move a cut.

        Three ways to fail: too little agreement with the audio, a tempo outside
        the range the detector searched (which means it clamped), or too few
        beats to establish a period at all.
        """
        return (
            self.confidence >= min_confidence
            and MIN_BPM <= self.bpm <= MAX_BPM
            and self.beat_count >= 2
        )

    # ------------------------------------------------------------ queries
    def nearest_beat_ms(self, position_ms: int) -> int:
        """The beat closest to a position. Ties resolve to the earlier beat.

        Linear rather than bisecting: a grid is a few hundred entries and this
        runs a handful of times per plan, so the simpler code is the right
        trade. Ties resolving early is arbitrary but fixed, which is what keeps
        two runs over the same input identical.
        """
        if not self.beats_ms:
            return position_ms
        best = self.beats_ms[0]
        best_delta = abs(best - position_ms)
        for beat in self.beats_ms[1:]:
            delta = abs(beat - position_ms)
            if delta < best_delta:
                best, best_delta = beat, delta
            elif beat > position_ms:
                # Ascending, so every later beat is further away still.
                break
        return best

    def beats_between(self, start_ms: int, end_ms: int) -> tuple[int, ...]:
        """Beats within a half-open window. Used by the editor to draw markers."""
        return tuple(b for b in self.beats_ms if start_ms <= b < end_ms)

    # ------------------------------------------------------------ snapping
    def span_for_beats(self, count: int) -> int:
        """How far ``count`` beats reach, measured from a beat.

        The one conversion from beats to milliseconds, and the reason it takes a
        *cumulative* count rather than a per-clip one: a beat period is rarely a
        whole number of milliseconds (128 BPM is 468.75), so rounding each clip
        separately and adding the results accumulates error at up to half a
        millisecond per cut. Rounding the running total instead keeps every cut
        within half a millisecond of its beat no matter how many precede it.

        So a clip's length is always the *difference of two of these*, never a
        product::

            duration = span_for_beats(placed + n) - span_for_beats(placed)
        """
        period = self.period_ms
        if period <= 0 or count <= 0:
            return 0
        return int(round(count * period))

    def snap_beats(self, target_ms: int, *, min_ms: int, max_ms: int) -> int | None:
        """The whole number of beats closest to ``target_ms`` that is renderable.

        Returns the *count*, not a duration. Callers need the integer: it is
        what goes in the plan's metadata, and it is what lets a running total be
        kept exactly. Handing back milliseconds and expecting the caller to
        divide is what makes a "whole number of beats" come back as 10.999.

        ``None`` means no whole-beat span works -- either none fits the bounds,
        or the nearest that does is further from the target than
        ``MAX_SNAP_DRIFT_RATIO`` allows. Quantising is a preference, not a
        requirement: a 25-second cut that becomes 40 seconds because the tempo
        is slow is not the edit anybody asked for.
        """
        period = self.period_ms
        if period <= 0 or not self.is_reliable():
            return None

        wanted = max(1, round(target_ms / period))
        limit = max(1.0, target_ms * MAX_SNAP_DRIFT_RATIO)

        for beats in _search_order(wanted):
            span = self.span_for_beats(beats)
            if span < min_ms or span > max_ms:
                continue
            if abs(span - target_ms) > limit:
                continue
            return beats
        return None

    def snap_span_ms(self, target_ms: int, *, min_ms: int, max_ms: int) -> int:
        """A single clip length quantised to whole beats, or the clamped target.

        A convenience over ``snap_beats`` for callers placing exactly one span.
        Anything laying out a *sequence* must use ``span_for_beats`` against a
        running beat count instead, or it will accumulate the rounding error
        this method cannot avoid on its own.
        """
        beats = self.snap_beats(target_ms, min_ms=min_ms, max_ms=max_ms)
        if beats is None:
            return max(min_ms, min(max_ms, target_ms))
        return self.span_for_beats(beats)

    # --------------------------------------------------------- persistence
    def as_payload(self) -> dict[str, Any]:
        return {
            "bpm": round(self.bpm, 3),
            "confidence": round(self.confidence, 4),
            "beat_count": self.beat_count,
            "beats_ms": list(self.beats_ms),
            "source_duration_ms": self.source_duration_ms,
        }


def _search_order(centre: int) -> list[int]:
    """Whole-beat counts to try, nearest the wanted count first.

    Outward from the centre so that the first candidate satisfying the bounds is
    also the closest to what was asked for. Bounded, because an unbounded search
    on a pathological tempo would walk to zero one beat at a time.
    """
    order = [centre]
    for step in range(1, 33):
        if centre - step >= 1:
            order.append(centre - step)
        order.append(centre + step)
    return order


def grid_from_payload(payload: dict[str, Any] | None) -> BeatGrid | None:
    """Rebuild a grid from a stored analysis payload.

    Returns ``None`` for anything unusable rather than raising. A malformed or
    missing beats row means "plan without beats", which is a supported outcome;
    turning it into an exception would make a bad analysis break planning
    entirely, which is a worse failure than an unsynced edit.
    """
    if not payload:
        return None
    try:
        raw_beats = payload.get("beats_ms") or []
        beats = tuple(sorted({int(b) for b in raw_beats if int(b) >= 0}))[:MAX_BEATS]
        duration = payload.get("source_duration_ms")
        return BeatGrid(
            bpm=float(payload.get("bpm", 0.0)),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
            beats_ms=beats,
            source_duration_ms=int(duration) if duration is not None else None,
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "MAX_BEATS",
    "MAX_BPM",
    "MAX_SNAP_DRIFT_RATIO",
    "MIN_BEAT_CONFIDENCE",
    "MIN_BPM",
    "BeatGrid",
    "grid_from_payload",
]
