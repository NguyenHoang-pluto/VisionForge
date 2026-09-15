"""How much the reference is allowed to change the edit.

    named style preset  ─┐
                         ├─ blend(strength) ─> StylePolicy ─> planner
    measured reference  ─┘

The policy is the only thing the planner sees. It exists so that "make it feel
like this reference" is a *dial* rather than a mode: at 0 the planner behaves
exactly as it did in Phase 4-7, at 100 the reference's measurements dominate the
pacing and the ranking, and the steps between are a straight interpolation with
no cliff.

Three properties are load-bearing.

**Zero is exactly the old behaviour.** Not approximately, not "close enough":
``blend(preset, reference, ZERO)`` returns the preset's own numbers and no
affinity term, so a plan made at strength 0 with a reference attached is the
same plan as one made with no reference at all. A test pins it, because the
moment that stops being true, every edit in the product has quietly changed.

**Confidence gates influence.** Each measurement's own confidence multiplies the
strength before it is applied, so a shot length read from three cuts moves the
pacing about a third as far as one read from twenty. A user who sets 100% is
asking for as much of the reference as the reference actually supports, not for
a confident answer to be invented.

**The reference never supplies footage.** A policy carries numbers -- target
durations, weights, a look vector. It has no media id for the reference and no
way to name one. What the reference contributes is how the user's *own* clips
are ranked and cut, which is the only meaning of "style" this product will ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from visionforge.domain.reference import Measurement, ReferenceProfile
from visionforge.domain.selection import SelectionWeights
from visionforge.domain.style import EditStyle, StyleProfile, profile_for

#: The share of a candidate's score that style affinity may take at full
#: strength.
#:
#: A third, and no more. Selection's first job is to reject footage nobody can
#: watch -- soft, crushed, blown -- and a style term that could outvote
#: sharpness would let a reference's palette put an out-of-focus shot in the
#: edit. Style decides between usable clips; it does not decide what usable
#: means.
MAX_AFFINITY = 0.35


class StyleStrength(StrEnum):
    """How much reference, as a closed set.

    A dial with five stops rather than a free number: the difference between 62%
    and 68% is not a decision anyone can make deliberately, and a closed set is
    a validated one at the API boundary.
    """

    ZERO = "0"
    QUARTER = "25"
    HALF = "50"
    THREE_QUARTER = "75"
    FULL = "100"

    @property
    def fraction(self) -> float:
        return int(self.value) / 100.0


@dataclass(frozen=True, slots=True)
class StyleTarget:
    """The look to steer towards, as normalised 0..1 axes.

    Only axes that were actually measured on the reference appear, and a
    candidate is compared on the axes it shares with the target. A candidate
    missing an axis is not penalised for it -- it is scored on the rest, because
    an unanalysed clip should not be ranked below an analysed one on the
    strength of the analysis being absent.
    """

    luminance: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    motion: float | None = None

    @property
    def axes(self) -> dict[str, float]:
        return {
            name: value
            for name, value in (
                ("luminance", self.luminance),
                ("contrast", self.contrast),
                ("saturation", self.saturation),
                ("motion", self.motion),
            )
            if value is not None
        }

    def __bool__(self) -> bool:
        return bool(self.axes)

    def affinity(self, **measured: float | None) -> float | None:
        """How close a candidate is to this target. 1.0 is identical.

        Mean absolute distance over the shared axes, subtracted from one. Mean
        rather than Euclidean: every axis is already 0..1 and none of them is
        more important than another, so the simpler measure is the honest one
        and its result is legible in a score breakdown.
        """
        shared = [
            (target, float(value))
            for name, target in self.axes.items()
            if isinstance(value := measured.get(name), int | float)
        ]
        if not shared:
            return None
        distance = sum(abs(target - value) for target, value in shared) / len(shared)
        return round(max(0.0, 1.0 - distance), 6)


@dataclass(frozen=True, slots=True)
class StylePolicy:
    """What the planner should do, after the reference has had its say."""

    strength: StyleStrength
    #: The named style this was blended from, if the user picked one.
    style: EditStyle | None
    #: Ranking weights, already blended. Their components still sum to one.
    weights: SelectionWeights
    min_clip_ms: int
    max_clip_ms: int
    target_clip_ms: int
    prefer_sequence: bool
    #: The look to steer towards. Empty when there is no usable reference.
    target: StyleTarget
    #: Advisory only. The reference having been cut to its music is a reason to
    #: offer beat sync, never a reason to switch it on behind the user: adding
    #: music and re-timing an edit stayed separate decisions in Phase 7 and stay
    #: separate here.
    suggests_beat_sync: bool = False
    #: Which reference measurements actually moved anything, and by how much.
    #: Recorded on the plan so an edit can be explained after the fact.
    influence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_clip_ms > self.max_clip_ms:
            raise ValueError("min_clip_ms cannot exceed max_clip_ms")

    def clamp_clip_ms(self, value: int) -> int:
        return max(self.min_clip_ms, min(self.max_clip_ms, value))

    @property
    def is_styled(self) -> bool:
        """Whether the reference is doing anything at all."""
        return self.strength is not StyleStrength.ZERO and bool(self.influence)

    def as_payload(self) -> dict[str, Any]:
        return {
            "strength": self.strength.value,
            "style": self.style.value if self.style else None,
            "target_clip_ms": self.target_clip_ms,
            "min_clip_ms": self.min_clip_ms,
            "max_clip_ms": self.max_clip_ms,
            "affinity_weight": round(self.weights.affinity, 4),
            "look": {name: round(value, 4) for name, value in self.target.axes.items()},
            "suggests_beat_sync": self.suggests_beat_sync,
            "influence": {k: round(v, 3) for k, v in sorted(self.influence.items())},
        }


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


def _pull(measurement: Measurement | None, strength: float) -> float:
    """How far a single measurement gets to pull, 0..1.

    The dial and the evidence, multiplied. Absent evidence pulls nothing.
    """
    if measurement is None:
        return 0.0
    return max(0.0, min(1.0, strength * measurement.confidence))


def blend(
    preset: StyleProfile,
    reference: ReferenceProfile | None,
    strength: StyleStrength,
) -> StylePolicy:
    """Combine a named style with a measured reference at the chosen strength.

    With no reference, an unusable one, or zero strength, this returns the
    preset unchanged and no affinity -- which is the Phase 4-7 planner exactly.
    """
    base = StylePolicy(
        strength=strength,
        style=preset.style if preset.style is not EditStyle.CUSTOM else preset.style,
        weights=preset.weights,
        min_clip_ms=preset.min_clip_ms,
        max_clip_ms=preset.max_clip_ms,
        target_clip_ms=preset.target_clip_ms,
        prefer_sequence=preset.prefer_sequence_order,
        target=StyleTarget(),
        influence={},
    )

    if reference is None or not reference.is_usable or strength is StyleStrength.ZERO:
        return base

    fraction = strength.fraction
    influence: dict[str, float] = {}

    # ------------------------------------------------------------- pacing
    #
    # The reference's median shot length pulls the target, and the bounds move
    # with it so that a pacing the style would have refused is not clamped
    # straight back to where it started. The plan-level limits still apply
    # afterwards and unconditionally -- this widens a style, never the renderer.
    target_ms = preset.target_clip_ms
    min_ms, max_ms = preset.min_clip_ms, preset.max_clip_ms

    pull = _pull(reference.shot_ms, fraction)
    if pull > 0 and reference.shot_ms is not None:
        measured = reference.shot_ms.value
        target_ms = int(round(_lerp(preset.target_clip_ms, measured, pull)))
        influence["shot_ms"] = round(pull, 3)

        # Quartiles, where the reference has them, give the bounds a shape that
        # is the reference's own rather than the preset's scaled. Where it does
        # not, the bounds follow the target by the same proportion.
        low = float(reference.shot_ms_p25 or measured * 0.6)
        high = float(reference.shot_ms_p75 or measured * 1.6)
        min_ms = int(round(_lerp(preset.min_clip_ms, low, pull)))
        max_ms = int(round(_lerp(preset.max_clip_ms, high, pull)))

        # An inverted or degenerate window is a blend artefact, not a style.
        min_ms = max(1, min(min_ms, target_ms))
        max_ms = max(target_ms, max_ms)

    # ------------------------------------------------------------ the look
    axes: dict[str, float] = {}
    pulls: list[float] = []
    for name, measurement in (
        ("luminance", reference.luminance),
        ("contrast", reference.contrast),
        ("saturation", reference.saturation),
        ("motion", reference.motion),
    ):
        axis_pull = _pull(measurement, fraction)
        if axis_pull > 0 and measurement is not None:
            axes[name] = measurement.value
            pulls.append(axis_pull)
            influence[name] = round(axis_pull, 3)

    target = StyleTarget(**axes)
    weights = preset.weights
    if target and pulls:
        # One affinity weight for the whole look, scaled by the average
        # confidence of the axes that made it up, and taken *out of* the
        # existing weights rather than added on top -- so the score stays in
        # 0..1 and the five usability components keep their relative balance.
        affinity = MAX_AFFINITY * (sum(pulls) / len(pulls))
        weights = _with_affinity(preset.weights, affinity)

    # ----------------------------------------------------------- the rhythm
    suggests_beat_sync = False
    if reference.beat_sync is not None:
        beat_pull = _pull(reference.beat_sync, fraction)
        if beat_pull > 0 and reference.beat_sync.value >= 0.5:
            suggests_beat_sync = True
            influence["beat_sync"] = round(beat_pull, 3)

    return replace(
        base,
        weights=weights,
        min_clip_ms=min_ms,
        max_clip_ms=max_ms,
        target_clip_ms=target_ms,
        target=target,
        suggests_beat_sync=suggests_beat_sync,
        influence=influence,
    )


def _with_affinity(weights: SelectionWeights, affinity: float) -> SelectionWeights:
    """Make room for the affinity term by scaling the others down.

    The five usability components are reduced by ``1 - affinity`` so their sum
    plus the new term is what it was. Their *ratios* are untouched: a style
    reference changes how much style matters, not whether sharpness matters more
    than duration.
    """
    share = max(0.0, min(MAX_AFFINITY, affinity))
    keep = 1.0 - share
    return replace(
        weights,
        sharpness=weights.sharpness * keep,
        exposure=weights.exposure * keep,
        contrast=weights.contrast * keep,
        resolution=weights.resolution * keep,
        duration=weights.duration * keep,
        affinity=share,
    )


def policy_for(
    style: EditStyle | None,
    reference: ReferenceProfile | None = None,
    strength: StyleStrength = StyleStrength.ZERO,
) -> StylePolicy:
    """The policy for a request. The one entry point the application layer needs."""
    return blend(profile_for(style), reference, strength)


__all__ = [
    "MAX_AFFINITY",
    "StylePolicy",
    "StyleStrength",
    "StyleTarget",
    "blend",
    "policy_for",
]
