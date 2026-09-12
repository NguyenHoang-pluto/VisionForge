"""Media analysis domain model.

Describes *what* an analysis is, never how it is computed. No OpenCV, no PyTorch,
no file I/O -- enforced by the ``domain-is-pure`` import-linter contract.

Two ideas carry most of the weight here:

- **Analyzers are versioned.** A result is identified by
  ``(media_id, analyzer, analyzer_version)``. Upgrading a model writes a new row
  rather than overwriting history, so a regression is visible instead of silent.
- **"Unsupported" is a success.** An audio file has no blur score. That is a fact
  about the medium, not a failure, and it must not fail a job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind


class AnalyzerName(StrEnum):
    """Registered analyzers. The string is persisted, so values are stable."""

    QUALITY = "quality"
    SCENES = "scenes"
    PHASH = "phash"
    CLIP = "clip"
    FACES = "faces"


class AnalyzerKind(StrEnum):
    """Which worker queue an analyzer belongs to."""

    CPU = "cpu"
    GPU = "gpu"


class AnalysisStatus(StrEnum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


# --------------------------------------------------------------------- inputs
@dataclass(frozen=True, slots=True)
class AnalysisSource:
    """A decoded-ready input for an analyzer.

    ``local_path`` is resolved by the worker, never by the caller: no client
    string reaches it. ``used_proxy`` records which representation was analysed,
    because a score computed on a 720p proxy is not interchangeable with one
    computed on a 4K original and the stored result says which it was.
    """

    media_id: MediaId
    kind: MediaKind
    local_path: str
    used_proxy: bool
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """What an analyzer returns.

    ``payload`` is analyzer-specific and lands in JSONB. ``embedding`` is broken
    out because it goes to a typed pgvector column, not into the JSON blob.
    """

    analyzer: AnalyzerName
    version: str
    status: AnalysisStatus
    payload: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status is AnalysisStatus.OK


@runtime_checkable
class Analyzer(Protocol):
    """The analyzer port.

    Application code depends on this and never on OpenCV, PyTorch or a model
    name. That is what lets the whole analysis pipeline be tested with fakes on a
    machine with no GPU.
    """

    @property
    def name(self) -> AnalyzerName: ...

    @property
    def version(self) -> str: ...

    @property
    def kind(self) -> AnalyzerKind: ...

    def supports(self, media_kind: MediaKind) -> bool: ...

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome: ...


# ---------------------------------------------------------------- frame sampling
#: Deterministic sample points through a video, as fractions of its duration.
#:
#: Not 0.0 and 1.0: the first and last frames of real footage are very often
#: black, a fade, or a slate, and sampling them would systematically bias every
#: downstream quality and embedding score. 5%/95% keeps the coverage without the
#: artefacts.
FRAME_SAMPLE_FRACTIONS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def sample_timestamps_ms(duration_ms: int | None) -> tuple[int, ...]:
    """Deterministic representative timestamps for a video.

    Same duration in, same timestamps out, always -- which is what makes an
    analysis reproducible and a regression attributable to the model rather than
    to frame luck.
    """
    if not duration_ms or duration_ms <= 0:
        return (0,)
    stamps = sorted({int(duration_ms * f) for f in FRAME_SAMPLE_FRACTIONS})
    return tuple(min(t, max(duration_ms - 1, 0)) for t in stamps)


# ------------------------------------------------------------------ thresholds
@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Tunable cut-offs for the quality analyzer.

    Collected in one object rather than scattered as literals, so they can be
    changed per-project later without hunting through the pipeline, and so the
    values used are recorded alongside the scores.
    """

    #: Laplacian variance below this reads as soft/out-of-focus. Scale depends on
    #: resolution, so the analyzer normalises before comparing.
    blur_min_variance: float = 100.0
    #: Luma (0-255) below/above which a pixel counts as crushed or blown.
    underexposed_below: int = 16
    overexposed_above: int = 239
    #: Fraction of pixels beyond those bounds that makes a frame badly exposed.
    max_clipped_ratio: float = 0.15
    #: Acceptable mean-luminance band for a well-exposed frame.
    good_luma_range: tuple[int, int] = (60, 200)
    #: Standard deviation of luma below this reads as flat/low-contrast.
    contrast_min_stddev: float = 25.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_clipped_ratio <= 1.0:
            raise ValueError("max_clipped_ratio must be a fraction between 0 and 1")
        if self.underexposed_below >= self.overexposed_above:
            raise ValueError("underexposed_below must be lower than overexposed_above")


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


# ----------------------------------------------------------- perceptual hashing
#: Hamming distance at or below which two 64-bit pHashes are near-duplicates.
#: 64-bit pHash: 0 is identical, <=5 survives resize and re-compression, >10
#: starts admitting merely-similar images.
PHASH_DUPLICATE_MAX_DISTANCE = 5
PHASH_BITS = 64


def hamming_distance(left: str, right: str) -> int:
    """Bit distance between two hex-encoded perceptual hashes."""
    if len(left) != len(right):
        raise ValueError("perceptual hashes must be the same length")
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def are_near_duplicates(
    left: str, right: str, *, max_distance: int = PHASH_DUPLICATE_MAX_DISTANCE
) -> bool:
    return hamming_distance(left, right) <= max_distance


# ------------------------------------------------------------------- embeddings
#: OpenCLIP ViT-B/32 output width. Pinned here because the pgvector column type
#: must match it; changing the model means a migration, not a config edit.
CLIP_EMBEDDING_DIM = 512
