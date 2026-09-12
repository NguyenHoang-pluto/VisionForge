"""Perceptual hashing for near-duplicate detection.

Phase 2 deduplicates on SHA-256 -- byte-identical files only. That misses the
case this module exists for: the same photo resized, re-compressed, or slightly
recoloured, which a phone gallery or a shared album produces constantly.

The classic DCT pHash is implemented directly on OpenCV's ``cv2.dct`` rather than
pulled in from ``imagehash``. It is roughly fifteen lines, it avoids a dependency
whose transitive requirements overlap what is already installed, and -- most
usefully -- the implementation is visible, so the hash is explainable rather than
a black box whose behaviour has to be taken on trust.

**This module only reports.** Nothing is deleted. Clusters are recorded and a
later selection stage decides what to keep.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from visionforge.domain.analysis import (
    PHASH_DUPLICATE_MAX_DISTANCE,
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
    hamming_distance,
)
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import sample_frames, to_grayscale

PHASH_ANALYZER_VERSION = "1"

#: The image is reduced to 32x32, DCT'd, and the top-left 8x8 block of low
#: frequencies is kept -- 64 bits. High frequencies are exactly the detail that
#: resizing and JPEG compression destroy, so discarding them is what makes the
#: hash survive those operations.
_DCT_SIZE = 32
_HASH_SIDE = 8


def perceptual_hash(gray: NDArray[np.uint8]) -> str:
    """64-bit DCT perceptual hash, hex-encoded.

    The DC coefficient is dropped rather than hashed. It is the image's average
    brightness -- a large positive number for any real image -- so it is always
    above the median of the AC terms and its bit is therefore always 1. Hashing
    it would waste one of 64 bits and guarantee that any two unrelated images
    agree on at least one. The 63 AC coefficients are thresholded against their
    own median and left-padded by one bit.
    """
    resized = cv2.resize(gray, (_DCT_SIZE, _DCT_SIZE), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))
    ac_terms = coefficients[:_HASH_SIDE, :_HASH_SIDE].flatten()[1:]

    median = float(np.median(ac_terms.astype(np.float64)))
    value = 0
    for coefficient in ac_terms:
        value = (value << 1) | int(coefficient > median)
    return f"{value:016x}"


def average_hash(gray: NDArray[np.uint8]) -> str:
    """8x8 mean-threshold hash. Cheaper and blunter than pHash.

    Stored alongside pHash as a second opinion: aHash is more sensitive to
    gamma/exposure shifts, so agreement between the two is a stronger duplicate
    signal than either alone.
    """
    resized = cv2.resize(gray, (_HASH_SIDE, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    value = 0
    for pixel in resized.flatten():
        value = (value << 1) | int(pixel > mean)
    return f"{value:016x}"


class PerceptualHashAnalyzer:
    """Perceptual hashes for images and representative video frames."""

    name = AnalyzerName.PHASH
    version = PHASH_ANALYZER_VERSION
    kind = AnalyzerKind.CPU

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind in (MediaKind.IMAGE, MediaKind.VIDEO)

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"perceptual hashing does not apply to {source.kind.value}"},
            )

        frames = sample_frames(source)
        per_frame = [
            {
                "timestamp_ms": frame.timestamp_ms,
                "phash": perceptual_hash(to_grayscale(frame.image)),
                "ahash": average_hash(to_grayscale(frame.image)),
            }
            for frame in frames
        ]

        # The first sampled frame is the asset's representative hash. For an
        # image there is only one; for a video, 5% in is far enough past the
        # opening fade to be characteristic.
        primary = per_frame[0]
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload={
                "phash": primary["phash"],
                "ahash": primary["ahash"],
                "frames": per_frame,
                "hash_bits": 64,
                "duplicate_max_distance": PHASH_DUPLICATE_MAX_DISTANCE,
                "used_proxy": source.used_proxy,
            },
        )


def cluster_by_hash(
    hashes: dict[str, str], *, max_distance: int = PHASH_DUPLICATE_MAX_DISTANCE
) -> list[list[str]]:
    """Group media ids whose hashes are within ``max_distance`` of each other.

    Single-link agglomeration over an O(n^2) comparison. That is fine for a
    project-sized batch (hundreds of assets, a hex XOR each) and keeps the
    behaviour obvious; an ANN index over hash space is a Phase 4 concern if
    projects ever reach the tens of thousands.

    Returns only groups of two or more -- a cluster of one is not a duplicate.
    """
    ids = list(hashes)
    parent = {media_id: media_id for media_id in ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if hamming_distance(hashes[left], hashes[right]) <= max_distance:
                union(left, right)

    groups: dict[str, list[str]] = {}
    for media_id in ids:
        groups.setdefault(find(media_id), []).append(media_id)

    return [sorted(group) for group in groups.values() if len(group) > 1]
