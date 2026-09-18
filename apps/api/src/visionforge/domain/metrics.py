"""Explainable metrics for a finished editorial plan.

Eight measurements, each with its own name, its own scale and its own sample
size. **They are deliberately not combined into a score.**

A single number would be the most requested feature here and the least
defensible one. "Edit quality: 0.72" cannot be acted on: it does not say whether
the problem is repetition or pacing, it hides a catastrophic beat alignment
behind an excellent diversity, and the weights that produced it would be an
invented claim about how much a repeated shot costs relative to an off-beat cut
-- a claim nobody can justify and everybody would then tune against. Eight
numbers can each be argued with, which is what makes them worth having.

Each metric reports whether it could be measured at all. ``beat_alignment`` over
an edit with no music is not zero, it is *absent*, and the difference is the same
one ``ReferenceProfile`` insists on: a measurement that was not taken is not a
measurement of failure.

Every metric's direction is stated with it, because half of them are better high
and half are better low, and a reader should not have to remember which.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from visionforge.domain.decisions import EditorialPlan
from visionforge.domain.editorial import SignalBoard
from visionforge.domain.ids import MediaId
from visionforge.domain.story import EditorialPolicy, StoryRole


class Direction(StrEnum):
    """Which way is better. Stated per metric rather than assumed."""

    HIGHER = "higher_is_better"
    LOWER = "lower_is_better"


#: Every metric's name and which way is better, declared once.
#:
#: The constructors below read their direction from here rather than repeating a
#: literal, and the API serves this same mapping to the editor. Three places
#: independently asserting that repetition is better low is three places to get
#: it wrong; one table is one.
METRIC_DIRECTIONS: dict[str, Direction] = {
    "content_diversity": Direction.HIGHER,
    "repetition": Direction.LOWER,
    "pacing_consistency": Direction.HIGHER,
    "energy_progression": Direction.HIGHER,
    "beat_alignment": Direction.HIGHER,
    "style_adherence": Direction.HIGHER,
    "story_completeness": Direction.HIGHER,
    "quality": Direction.HIGHER,
}


@dataclass(frozen=True, slots=True)
class Metric:
    """One measurement, its direction, and how much evidence it rests on.

    ``value`` is ``None`` when the metric could not be measured. That is a
    first-class outcome, not a zero: an edit with no music has no beat
    alignment, and reporting 0.0 would read as "every cut missed".
    """

    name: str
    value: float | None
    direction: Direction
    #: How many observations the value rests on. A diversity measured over two
    #: clips is a weaker claim than one over twelve, and the number says so.
    sample_size: int = 0
    #: Why it could not be measured, when it could not. A short code, not prose.
    unmeasurable: str | None = None

    @property
    def measured(self) -> bool:
        return self.value is not None

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": None if self.value is None else round(self.value, 4),
            "direction": self.direction.value,
            "sample_size": self.sample_size,
            "unmeasurable": self.unmeasurable,
        }


def _metric(
    name: str,
    value: float | None,
    *,
    sample_size: int = 0,
    unmeasurable: str | None = None,
) -> Metric:
    """Build a metric, taking its direction from the one table that knows it.

    The value is clamped to 0..1 here rather than at each call site. Cosine
    similarity runs -1..1, so a diversity computed from it can exceed one, and a
    number outside the scale a metric declares is worse than a saturated one.
    """
    clamped = None if value is None else round(max(0.0, min(1.0, value)), 4)
    return Metric(
        name=name,
        value=clamped,
        direction=METRIC_DIRECTIONS[name],
        sample_size=sample_size,
        unmeasurable=unmeasurable,
    )


@dataclass(frozen=True, slots=True)
class EditMetrics:
    """The eight measurements, kept apart.

    Iterable and addressable by name, because the UI wants a list and the tests
    want a lookup, and neither should have to know the field order.
    """

    content_diversity: Metric
    repetition: Metric
    pacing_consistency: Metric
    energy_progression: Metric
    beat_alignment: Metric
    style_adherence: Metric
    story_completeness: Metric
    quality: Metric

    @property
    def all(self) -> tuple[Metric, ...]:
        return (
            self.content_diversity,
            self.repetition,
            self.pacing_consistency,
            self.energy_progression,
            self.beat_alignment,
            self.style_adherence,
            self.story_completeness,
            self.quality,
        )

    def get(self, name: str) -> Metric | None:
        return next((metric for metric in self.all if metric.name == name), None)

    @property
    def measured_count(self) -> int:
        return sum(1 for metric in self.all if metric.measured)

    def as_payload(self) -> dict[str, Any]:
        # A mapping keyed by name, so a consumer reads ``metrics["repetition"]``
        # rather than indexing a list whose order it would have to trust.
        return {metric.name: metric.as_payload() for metric in self.all}


def evaluate(plan: EditorialPlan, board: SignalBoard, policy: EditorialPolicy) -> EditMetrics:
    """Measure a finished editorial plan. Pure, deterministic, and total."""
    ids = tuple(segment.media_id for segment in plan.segments)
    return EditMetrics(
        content_diversity=_diversity(board, ids),
        repetition=_repetition(board, ids),
        pacing_consistency=_pacing_consistency(plan),
        energy_progression=_energy_progression(plan),
        beat_alignment=_beat_alignment(plan),
        style_adherence=_style_adherence(plan),
        story_completeness=_story_completeness(plan, policy),
        quality=_quality(plan),
    )


# ------------------------------------------------------------------- content
def _pairs(board: SignalBoard, ids: tuple[MediaId, ...]) -> list[float]:
    """Every pairwise similarity that could be computed."""
    scores: list[float] = []
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            score = board.similarity(left, right)
            if score is not None:
                scores.append(score)
    return scores


def _diversity(board: SignalBoard, ids: tuple[MediaId, ...]) -> Metric:
    """Mean semantic distance between the chosen clips.

    The *mean*, which measures the edit's overall spread. It is paired with
    ``repetition`` below, which measures its worst pair -- two questions a single
    number cannot answer at once, because an edit of eleven varied shots and two
    identical ones has excellent mean diversity and a real problem.
    """
    scores = _pairs(board, ids)
    if not scores:
        return _metric("content_diversity", None, unmeasurable="no_embeddings")
    mean = sum(scores) / len(scores)
    return _metric("content_diversity", 1.0 - mean, sample_size=len(scores))


def _repetition(board: SignalBoard, ids: tuple[MediaId, ...]) -> Metric:
    """The closest pair in the edit. Lower is better.

    Deliberately the maximum rather than an average: repetition is a property of
    the worst offender, and averaging it away is how an edit ships with the same
    shot in it twice.
    """
    scores = _pairs(board, ids)
    if not scores:
        return _metric("repetition", None, unmeasurable="no_embeddings")
    return _metric("repetition", max(scores), sample_size=len(scores))


# -------------------------------------------------------------------- rhythm
def _pacing_consistency(plan: EditorialPlan) -> Metric:
    """How closely the realised clip lengths track the ones pacing asked for.

    Low here is not automatically a failure: a source clip too short to fill its
    slot legitimately pulls the realised length away from the intended one. It
    *is* the number that says whether the curve survived contact with the
    footage, which is exactly what nobody could see before this phase.
    """
    slots = plan.pacing.slots
    segments = plan.segments
    count = min(len(slots), len(segments))
    if count == 0:
        return _metric("pacing_consistency", None, unmeasurable="no_segments")

    errors: list[float] = []
    for index in range(count):
        intended = slots[index].duration_ms
        if intended <= 0:
            continue
        errors.append(min(1.0, abs(segments[index].output_ms - intended) / intended))
    if not errors:
        return _metric("pacing_consistency", None, unmeasurable="no_intended_durations")

    return _metric("pacing_consistency", 1.0 - sum(errors) / len(errors), sample_size=len(errors))


def _energy_progression(plan: EditorialPlan) -> Metric:
    """Whether the edit's energy moves the way the curve wanted it to.

    Measured as *direction agreement* between adjacent clips, not as level
    match, and the distinction is the whole meaning of the word "progression".
    An edit whose clips are uniformly calmer than the curve asked for but which
    still accelerates into its peak has done the thing pacing is for; one whose
    levels happen to match but which gets quieter where the curve rises has not.

    Ties in the intended curve -- a ``STEADY`` shape, or two slots at the same
    energy -- are not counted either way. Asking whether a flat curve was
    followed upward is a question with no answer.
    """
    segments = plan.segments
    if len(segments) < 2:
        return _metric("energy_progression", None, unmeasurable="too_few_segments")

    agree = 0
    judged = 0
    for left, right in pairwise(segments):
        intended = right.target_energy - left.target_energy
        if abs(intended) < 1e-6:
            continue
        judged += 1
        realised = right.energy - left.energy
        if (intended > 0 and realised > 0) or (intended < 0 and realised < 0):
            agree += 1

    if judged == 0:
        return _metric("energy_progression", None, unmeasurable="flat_curve")
    return _metric("energy_progression", agree / judged, sample_size=judged)


def _beat_alignment(plan: EditorialPlan) -> Metric:
    """What share of the cuts landed on a beat.

    Absent, not zero, when the edit was never asked to be beat-synced or when
    the grid was not trusted -- the same distinction ``beat_sync_tendency`` draws
    for a reference video, and for the same reason: "not measured" and "measured
    and it missed" are different facts about an edit.
    """
    sync = plan.pacing.beat_sync
    if sync is None:
        return _metric("beat_alignment", None, unmeasurable="not_requested")
    if not sync.get("applied"):
        reason = sync.get("reason")
        return _metric(
            "beat_alignment", None, unmeasurable=str(reason) if reason else "not_applied"
        )

    total = len(plan.segments)
    if total == 0:
        return _metric("beat_alignment", None, unmeasurable="no_segments")
    on_beat = sum(1 for segment in plan.segments if segment.on_beat)
    return _metric("beat_alignment", on_beat / total, sample_size=total)


# ----------------------------------------------------------------------- look
def _style_adherence(plan: EditorialPlan) -> Metric:
    """How close the chosen clips are to the project's reference video.

    Absent when there is no reference, which is most projects. A style score of
    zero for an edit nobody asked to be styled would be a criticism of something
    that never happened -- and the ``style_match`` component defaults to 1.0
    when unmeasured, so without this gate a project with no reference would
    report perfect adherence to nothing.
    """
    styled = plan.intent is not None and bool(plan.intent.get("reference_style"))
    matches = [
        segment.components["style_match"]
        for segment in plan.segments
        if "style_match" in segment.components
    ]
    if not matches or not styled:
        return _metric("style_adherence", None, unmeasurable="no_reference")
    return _metric("style_adherence", sum(matches) / len(matches), sample_size=len(matches))


# ---------------------------------------------------------------------- story
def _story_completeness(plan: EditorialPlan, policy: EditorialPolicy) -> Metric:
    """How much of the policy's arc the footage actually supported.

    Weighted by each slot's priority, so losing the peak costs far more than
    losing the setup. An edit of three clips cannot carry a six-part arc and
    should not be marked down as though it failed at one -- what this reports is
    what the arc got, which is a fact about the footage as much as about the
    edit.
    """
    slots = policy.arc.slots
    if not slots:
        return _metric("story_completeness", None, unmeasurable="no_arc")

    present: set[StoryRole] = set(plan.roles)
    total = sum(slot.priority for slot in slots)
    if total <= 0:
        return _metric("story_completeness", None, unmeasurable="no_arc")
    covered = sum(slot.priority for slot in slots if slot.role in present)
    return _metric("story_completeness", covered / total, sample_size=len(slots))


def _quality(plan: EditorialPlan) -> Metric:
    """Mean technical quality of what was chosen.

    Still reported, and still one metric among eight rather than *the* metric.
    A high-quality edit of one repeated shot is a bad edit, and this number is
    where that becomes visible only when it is read next to ``repetition``.
    """
    scores = [
        segment.components["quality"]
        for segment in plan.segments
        if "quality" in segment.components
    ]
    if not scores:
        return _metric("quality", None, unmeasurable="no_scores")
    return _metric("quality", sum(scores) / len(scores), sample_size=len(scores))


__all__ = ["METRIC_DIRECTIONS", "Direction", "EditMetrics", "Metric", "evaluate"]
