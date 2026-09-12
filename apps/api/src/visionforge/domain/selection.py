"""Deterministic media selection and ranking.

**This is a heuristic scoring system, not taste.** It cannot tell a good shot
from a bad one. What it can do reliably is the thing a first pass actually needs:
discard the frames that are objectively unusable -- out of focus, crushed to
black, blown out, flat -- and avoid picking five copies of the same moment.

Everything here is a pure function of the analysis records. Same input, same
ranking, forever. That is what makes an automatically generated edit
reproducible, and a change in the output attributable to a weight change rather
than to chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visionforge.domain.analysis import (
    PHASH_DUPLICATE_MAX_DISTANCE,
    AnalyzerName,
    hamming_distance,
)
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind


class RejectionReason(StrEnum):
    """Why a candidate was excluded. Recorded so a selection can be explained."""

    NOT_VISUAL = "not_visual"
    NOT_READY = "not_ready"
    NO_ANALYSIS = "no_analysis"
    TOO_BLURRY = "too_blurry"
    BADLY_EXPOSED = "badly_exposed"
    LOW_CONTRAST = "low_contrast"
    TOO_SHORT = "too_short"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    """How much each signal contributes, and where the usability floor sits.

    Collected in one injectable object rather than scattered as literals, so a
    project can be re-planned with different priorities without touching the
    ranking code -- and so the weights that produced a given edit can be stored
    alongside it.
    """

    # --- contribution to the score, each normalised to 0..1 before weighting ---
    sharpness: float = 0.40
    exposure: float = 0.25
    contrast: float = 0.20
    resolution: float = 0.10
    duration: float = 0.05

    # --- usability floor: below these a candidate is rejected outright ---
    min_blur_score: float = 40.0
    max_clipped_ratio: float = 0.35
    min_contrast: float = 12.0
    min_duration_ms: int = 800

    # --- deduplication ---
    duplicate_max_distance: int = PHASH_DUPLICATE_MAX_DISTANCE

    #: Sharpness is unbounded above, so it is normalised against this ceiling
    #: rather than against the batch maximum. Normalising within a batch would
    #: make a clip's score depend on what it was uploaded alongside, which is
    #: exactly the kind of hidden coupling that makes results irreproducible.
    sharpness_reference: float = 600.0

    #: Likewise for resolution: 1080p is "full marks", not "the best in this set".
    resolution_reference_pixels: int = 1920 * 1080

    def __post_init__(self) -> None:
        total = self.sharpness + self.exposure + self.contrast + self.resolution + self.duration
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"selection weights must sum to 1.0, got {total}")
        if self.sharpness_reference <= 0 or self.resolution_reference_pixels <= 0:
            raise ValueError("normalisation references must be positive")


DEFAULT_SELECTION_WEIGHTS = SelectionWeights()


@dataclass(frozen=True, slots=True)
class Candidate:
    """One media asset reduced to the signals selection cares about.

    Built from ``media_analysis`` rows by the application layer, so this module
    stays free of the database.
    """

    media_id: MediaId
    kind: MediaKind
    is_ready: bool
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None

    # --- from the quality analyzer ---
    blur_score: float | None = None
    contrast: float | None = None
    mean_luminance: float | None = None
    clipped_ratio: float | None = None

    # --- from the pHash analyzer ---
    phash: str | None = None

    # --- from the scene analyzer ---
    scene_count: int | None = None

    #: Tie-break key. The upload order is stable and meaningful; a UUID is not.
    sequence: int = 0

    @property
    def has_quality(self) -> bool:
        return self.blur_score is not None and self.contrast is not None


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its score and the components that produced it.

    The breakdown is kept because a score without its parts cannot be argued
    with, and the whole point of a deterministic ranker is that it can be.
    """

    candidate: Candidate
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Rejection:
    media_id: MediaId
    reason: RejectionReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """What the selector decided, and why."""

    selected: tuple[ScoredCandidate, ...]
    rejected: tuple[Rejection, ...]
    duplicate_groups: tuple[tuple[MediaId, ...], ...] = ()

    @property
    def selected_ids(self) -> tuple[MediaId, ...]:
        return tuple(item.candidate.media_id for item in self.selected)


# --------------------------------------------------------------------- scoring
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sharpness_component(candidate: Candidate, weights: SelectionWeights) -> float:
    """Laplacian variance normalised against a fixed reference, not the batch."""
    if candidate.blur_score is None:
        return 0.0
    return _clamp(candidate.blur_score / weights.sharpness_reference)


def exposure_component(candidate: Candidate, weights: SelectionWeights) -> float:
    """Peaks at mid-grey and falls off toward either end of the range.

    Both directions are penalised: a crushed frame and a blown one are equally
    unusable, and a mean alone cannot tell them apart from a correct one.
    Clipping is subtracted separately because an image can average a perfect 128
    while being half black and half white.
    """
    if candidate.mean_luminance is None:
        return 0.0

    # 1.0 at luma 128, 0.0 at 0 or 255.
    balance = 1.0 - abs(candidate.mean_luminance - 128.0) / 128.0
    clipped = candidate.clipped_ratio or 0.0
    return _clamp(balance * (1.0 - _clamp(clipped)))


def contrast_component(candidate: Candidate, weights: SelectionWeights) -> float:
    """Luma standard deviation, normalised against a practical ceiling.

    Roughly 80 is a well-separated natural image; beyond that is usually a
    high-key graphic rather than better footage, so the curve saturates.
    """
    if candidate.contrast is None:
        return 0.0
    return _clamp(candidate.contrast / 80.0)


def resolution_component(candidate: Candidate, weights: SelectionWeights) -> float:
    if not candidate.width or not candidate.height:
        return 0.0
    pixels = candidate.width * candidate.height
    return _clamp(pixels / weights.resolution_reference_pixels)


def duration_component(candidate: Candidate, weights: SelectionWeights) -> float:
    """Mildly prefers clips with room to trim.

    Saturates at 10 s: past that, extra length says nothing about whether the
    material is good, only that there is more of it.
    """
    if not candidate.duration_ms:
        return 0.0
    return _clamp(candidate.duration_ms / 10_000.0)


def score_candidate(
    candidate: Candidate, weights: SelectionWeights = DEFAULT_SELECTION_WEIGHTS
) -> ScoredCandidate:
    """Weighted sum of the normalised components. Always in 0..1."""
    components = {
        "sharpness": sharpness_component(candidate, weights),
        "exposure": exposure_component(candidate, weights),
        "contrast": contrast_component(candidate, weights),
        "resolution": resolution_component(candidate, weights),
        "duration": duration_component(candidate, weights),
    }
    score = (
        components["sharpness"] * weights.sharpness
        + components["exposure"] * weights.exposure
        + components["contrast"] * weights.contrast
        + components["resolution"] * weights.resolution
        + components["duration"] * weights.duration
    )
    return ScoredCandidate(
        candidate=candidate,
        score=round(score, 6),
        components={k: round(v, 6) for k, v in components.items()},
    )


# ----------------------------------------------------------------- usability
def usability_rejection(
    candidate: Candidate, weights: SelectionWeights = DEFAULT_SELECTION_WEIGHTS
) -> Rejection | None:
    """Reject what is objectively unusable. ``None`` means it passed.

    A floor rather than a ranking: these are the failures no amount of good
    framing compensates for.
    """
    if candidate.kind is MediaKind.AUDIO:
        return Rejection(candidate.media_id, RejectionReason.NOT_VISUAL, "audio has no frames")
    if not candidate.is_ready:
        return Rejection(candidate.media_id, RejectionReason.NOT_READY, "ingest not finished")
    if not candidate.has_quality:
        return Rejection(
            candidate.media_id, RejectionReason.NO_ANALYSIS, "no quality analysis recorded"
        )

    assert candidate.blur_score is not None and candidate.contrast is not None

    # Exposure and contrast are checked *before* sharpness, and the order is the
    # point. A frame crushed to black or blown to white has almost no edge
    # energy, so its Laplacian variance is near zero and the blur test fires
    # first -- reporting "out of focus" for a frame whose real problem is that
    # it is black. Sharpness is only a meaningful measurement on a frame that
    # has detail to measure, so the checks that establish that run first.
    if (candidate.clipped_ratio or 0.0) > weights.max_clipped_ratio:
        return Rejection(
            candidate.media_id,
            RejectionReason.BADLY_EXPOSED,
            f"{(candidate.clipped_ratio or 0) * 100:.0f}% of pixels clipped",
        )
    if candidate.contrast < weights.min_contrast:
        return Rejection(
            candidate.media_id,
            RejectionReason.LOW_CONTRAST,
            f"contrast {candidate.contrast:.1f} below {weights.min_contrast}",
        )
    if candidate.blur_score < weights.min_blur_score:
        return Rejection(
            candidate.media_id,
            RejectionReason.TOO_BLURRY,
            f"sharpness {candidate.blur_score:.1f} below {weights.min_blur_score}",
        )
    # Images have no duration and are exempt; a video too short to trim is not
    # worth a cut.
    if candidate.kind is MediaKind.VIDEO and (candidate.duration_ms or 0) < weights.min_duration_ms:
        return Rejection(
            candidate.media_id,
            RejectionReason.TOO_SHORT,
            f"{candidate.duration_ms or 0} ms below {weights.min_duration_ms} ms",
        )
    return None


# ------------------------------------------------------------- deduplication
def group_duplicates(candidates: list[Candidate], *, max_distance: int) -> list[list[MediaId]]:
    """Cluster candidates whose perceptual hashes are within ``max_distance``.

    Single-link union-find over an O(n^2) comparison, same as the analysis-side
    clustering. Candidates without a hash are never grouped: an unknown hash is
    not evidence of similarity.

    Groups come back in stable order so that the survivor chosen downstream is
    deterministic.
    """
    hashed = [c for c in candidates if c.phash]
    parent: dict[MediaId, MediaId] = {c.media_id: c.media_id for c in hashed}

    def find(node: MediaId) -> MediaId:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index, left in enumerate(hashed):
        for right in hashed[index + 1 :]:
            assert left.phash and right.phash
            if hamming_distance(left.phash, right.phash) <= max_distance:
                root_left, root_right = find(left.media_id), find(right.media_id)
                if root_left != root_right:
                    parent[root_right] = root_left

    groups: dict[MediaId, list[MediaId]] = {}
    for candidate in hashed:
        groups.setdefault(find(candidate.media_id), []).append(candidate.media_id)

    order = {c.media_id: i for i, c in enumerate(candidates)}
    return [sorted(group, key=lambda m: order[m]) for group in groups.values() if len(group) > 1]


# ------------------------------------------------------------------- selector
def select(
    candidates: list[Candidate],
    *,
    limit: int,
    weights: SelectionWeights = DEFAULT_SELECTION_WEIGHTS,
) -> SelectionResult:
    """Rank, deduplicate and take the best ``limit`` candidates.

    Order of operations matters. Usability is checked first so that a blurry
    frame can never win its duplicate group; deduplication runs on the survivors
    so that the highest-scoring member of each group represents it; and the
    final ordering is by score with upload sequence as a tie-break, so two runs
    over identical input produce identical edits.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    rejections: list[Rejection] = []
    usable: list[Candidate] = []
    for candidate in candidates:
        rejection = usability_rejection(candidate, weights)
        if rejection is not None:
            rejections.append(rejection)
        else:
            usable.append(candidate)

    scored = {c.media_id: score_candidate(c, weights) for c in usable}
    groups = group_duplicates(usable, max_distance=weights.duplicate_max_distance)

    # Keep the best-scoring member of each duplicate group; reject the rest.
    suppressed: set[MediaId] = set()
    for group in groups:
        ranked = sorted(
            group,
            key=lambda m: (-scored[m].score, scored[m].candidate.sequence),
        )
        keeper = ranked[0]
        for media_id in ranked[1:]:
            suppressed.add(media_id)
            rejections.append(
                Rejection(
                    media_id,
                    RejectionReason.NEAR_DUPLICATE,
                    f"near-duplicate of {keeper}",
                )
            )

    survivors = [scored[c.media_id] for c in usable if c.media_id not in suppressed]
    survivors.sort(key=lambda s: (-s.score, s.candidate.sequence))

    return SelectionResult(
        selected=tuple(survivors[:limit]),
        rejected=tuple(rejections),
        duplicate_groups=tuple(tuple(g) for g in groups),
    )


def weights_payload(weights: SelectionWeights) -> dict[str, Any]:
    """The weights in force, for storing alongside a plan."""
    return {
        "sharpness": weights.sharpness,
        "exposure": weights.exposure,
        "contrast": weights.contrast,
        "resolution": weights.resolution,
        "duration": weights.duration,
        "min_blur_score": weights.min_blur_score,
        "max_clipped_ratio": weights.max_clipped_ratio,
        "min_contrast": weights.min_contrast,
        "min_duration_ms": weights.min_duration_ms,
        "duplicate_max_distance": weights.duplicate_max_distance,
    }


__all__ = [
    "DEFAULT_SELECTION_WEIGHTS",
    "AnalyzerName",
    "Candidate",
    "Rejection",
    "RejectionReason",
    "ScoredCandidate",
    "SelectionResult",
    "SelectionWeights",
    "group_duplicates",
    "score_candidate",
    "select",
    "usability_rejection",
    "weights_payload",
]
