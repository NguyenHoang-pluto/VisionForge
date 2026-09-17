"""Variants: the same footage, edited three different ways.

    policy  +  variant  ->  policy'

A variant is not a second edit of the first. It is a **modifier applied to the
policy before anything is decided**, so a variant changes which clips are chosen,
how many, how long each is held, where the energy peaks, and what treatment the
seams get. Everything downstream is the unmodified engine.

That is the difference between offering real alternatives and offering a shuffle.
Shuffling is what a variant system degenerates into when it is bolted on after
selection: the clips are already chosen, so the only freedom left is their order,
and three orders of the same six clips is one edit wearing three hats.

Three variants ship, and they are chosen to be *different in kind* rather than in
degree -- a faster version of the same edit is not an alternative anybody needs.
High energy re-selects toward movement and cuts hard; cinematic re-selects toward
composition, holds longer and dissolves; social fast cut front-loads the
strongest shot and discards the arc's preamble entirely.

Pure tables and a ``replace``. No I/O, no model, and no knowledge of what a
render is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from visionforge.domain.pacing import PacingShape
from visionforge.domain.story import ArcSlot, CreativeWeights, EditorialPolicy, StoryArc

#: Bumped when a variant's numbers change in a way that could alter an edit.
VARIANT_VERSION = "1"


class VariantId(StrEnum):
    """The alternatives a user may be offered. A closed set."""

    HIGH_ENERGY = "high_energy"
    CINEMATIC = "cinematic"
    SOCIAL_FAST_CUT = "social_fast_cut"


def _scaled_weights(base: CreativeWeights, **deltas: float) -> CreativeWeights:
    """Shift some weights and renormalise, keeping the sum at one.

    Additive shifts followed by a renormalisation, rather than direct
    assignment: a variant says "care more about energy" without having to know
    what the policy's other five numbers currently are, and the convex
    combination that makes two clips comparable survives.
    """
    values = {
        "quality": base.quality,
        "energy_fit": base.energy_fit,
        "diversity": base.diversity,
        "role_fit": base.role_fit,
        "style_match": base.style_match,
        "relevance": base.relevance,
    }
    for name, delta in deltas.items():
        values[name] = max(0.0, values[name] + delta)
    total = sum(values.values()) or 1.0
    return CreativeWeights(**{name: value / total for name, value in values.items()})


@dataclass(frozen=True, slots=True)
class Variant:
    """One named alternative, as a set of modifications to a policy."""

    id: VariantId
    label: str
    #: One line for the UI, and for the model as this variant's definition.
    description: str

    pacing: PacingShape
    pacing_spread: float
    #: Multiplier on the policy's clip-length bounds. Below 1 cuts faster.
    clip_scale: float = 1.0
    #: Added to the policy's diversity floor. A variant that cuts faster needs
    #: *more* diversity, not less: twenty short shots of one thing is a worse
    #: experience than six long ones of it.
    diversity_delta: float = 0.0
    transition_appetite: float = 0.2
    effect_appetite: float = 0.2
    slow_motion_peak: bool = False
    beat_sync_preference: float = 0.3
    #: Added to every arc slot's wanted energy, clamped to 0..1. This is what
    #: makes a variant *re-select* rather than re-time: a role that now wants
    #: 0.9 energy will pick a different clip.
    energy_bias: float = 0.0
    #: Multiplier on the clip budget. A variant may use more or fewer shots for
    #: the same length -- which is most of what makes it a different edit.
    #:
    #: There is deliberately no duration multiplier. A variant changes *how* the
    #: footage is cut, never how long the result runs: the user asked for a
    #: length, and silently returning a shorter video because they clicked
    #: "social" would be answering a question they did not ask.
    clip_budget_scale: float = 1.0
    #: Shifts applied to the creative weights before renormalisation.
    weight_deltas: tuple[tuple[str, float], ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": VARIANT_VERSION,
            "id": self.id.value,
            "label": self.label,
            "description": self.description,
            "pacing": self.pacing.value,
            "clip_scale": round(self.clip_scale, 3),
            "clip_budget_scale": round(self.clip_budget_scale, 3),
            "energy_bias": round(self.energy_bias, 3),
            "diversity_delta": round(self.diversity_delta, 3),
            "slow_motion_peak": self.slow_motion_peak,
            "beat_sync_preference": round(self.beat_sync_preference, 3),
        }


VARIANTS: dict[VariantId, Variant] = {
    VariantId.HIGH_ENERGY: Variant(
        id=VariantId.HIGH_ENERGY,
        label="High energy",
        description=(
            "Movement first. Shorter holds, harder cuts, the strongest moment "
            "slowed, and energy that only rises."
        ),
        pacing=PacingShape.BUILD,
        pacing_spread=0.65,
        clip_scale=0.62,
        diversity_delta=0.03,
        transition_appetite=0.05,
        effect_appetite=0.5,
        slow_motion_peak=True,
        beat_sync_preference=0.8,
        energy_bias=0.20,
        clip_budget_scale=1.35,
        weight_deltas=(("energy_fit", 0.10), ("quality", -0.04), ("style_match", -0.02)),
    ),
    VariantId.CINEMATIC: Variant(
        id=VariantId.CINEMATIC,
        label="Cinematic",
        description=(
            "Composition first. Long held takes, a calm opening, dissolves where "
            "the edit settles, and one moment that lifts."
        ),
        pacing=PacingShape.WAVE,
        pacing_spread=0.45,
        clip_scale=1.75,
        diversity_delta=0.05,
        transition_appetite=0.65,
        effect_appetite=0.35,
        slow_motion_peak=False,
        beat_sync_preference=0.1,
        energy_bias=-0.15,
        clip_budget_scale=0.6,
        weight_deltas=(("quality", 0.08), ("diversity", 0.04), ("energy_fit", -0.06)),
    ),
    VariantId.SOCIAL_FAST_CUT: Variant(
        id=VariantId.SOCIAL_FAST_CUT,
        label="Social fast cut",
        description=(
            "The best shot first and no preamble. Very short holds, a falling "
            "energy curve, cut to the music where there is any."
        ),
        pacing=PacingShape.DECAY,
        pacing_spread=0.5,
        clip_scale=0.5,
        diversity_delta=0.02,
        transition_appetite=0.0,
        effect_appetite=0.2,
        slow_motion_peak=False,
        beat_sync_preference=0.75,
        energy_bias=0.10,
        clip_budget_scale=1.6,
        weight_deltas=(("quality", 0.04), ("energy_fit", 0.06), ("relevance", -0.02)),
    ),
}


def _biased(arc: StoryArc, bias: float) -> StoryArc:
    """The same arc with every slot's wanted energy shifted.

    The *wanted* energy, not the emphasis or the weights: a variant changes what
    each role is looking for, which changes which clip wins it. Shifting the
    emphasis instead would only change how long the same clip is held, which is
    re-timing rather than re-editing.
    """
    if abs(bias) < 1e-9:
        return arc
    return StoryArc(
        name=arc.name,
        slots=tuple(
            ArcSlot(
                role=slot.role,
                weight=slot.weight,
                min_clips=slot.min_clips,
                max_clips=slot.max_clips,
                affinity=dict(slot.affinity),
                energy=max(0.0, min(1.0, slot.energy + bias)),
                emphasis=slot.emphasis,
                priority=slot.priority,
            )
            for slot in arc.slots
        ),
    )


def apply_variant(policy: EditorialPolicy, variant: Variant) -> EditorialPolicy:
    """The policy a variant asks for. Deterministic, and total.

    The clip bounds are scaled and then re-ordered, because a scale that pushes
    the target past the ceiling would otherwise produce an invalid policy -- and
    a policy that raises at construction is a variant that crashes a request
    rather than producing a different edit.
    """
    low = max(1, int(round(policy.min_clip_ms * variant.clip_scale)))
    target = max(1, int(round(policy.target_clip_ms * variant.clip_scale)))
    high = max(1, int(round(policy.max_clip_ms * variant.clip_scale)))
    low, high = min(low, high), max(low, high)
    target = max(low, min(high, target))

    return replace(
        policy,
        arc=_biased(policy.arc, variant.energy_bias),
        pacing=variant.pacing,
        pacing_spread=variant.pacing_spread,
        min_clip_ms=low,
        target_clip_ms=target,
        max_clip_ms=high,
        weights=_scaled_weights(policy.weights, **dict(variant.weight_deltas)),
        min_diversity=max(0.0, min(1.0, policy.min_diversity + variant.diversity_delta)),
        transition_appetite=variant.transition_appetite,
        effect_appetite=variant.effect_appetite,
        slow_motion_peak=variant.slow_motion_peak,
        beat_sync_preference=variant.beat_sync_preference,
    )


def variant_for(variant_id: VariantId | None) -> Variant | None:
    return None if variant_id is None else VARIANTS[variant_id]


def describe_variants() -> list[dict[str, Any]]:
    """The variant table for the API's capabilities endpoint."""
    return [variant.as_payload() for variant in VARIANTS.values()]


__all__ = [
    "VARIANTS",
    "VARIANT_VERSION",
    "Variant",
    "VariantId",
    "apply_variant",
    "describe_variants",
    "variant_for",
]
