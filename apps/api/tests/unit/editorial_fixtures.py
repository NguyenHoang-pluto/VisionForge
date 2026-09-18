"""Constructed footage for the editorial tests.

Every clip here is a plain ``Candidate``: no database, no media, no GPU, no
analyzer. The point of the editorial layer is that it is a pure function of the
analysis rows, and these fixtures are what that claim buys -- a football project
with a real peak in it, described in twenty lines and identical on every machine.

``vector`` deserves a note. CLIP embeddings are 512 floats; these are sixteen,
because nothing in the domain depends on the width and the tests are far easier
to read when two clips "about the same thing" share a seed. What matters is the
property the code relies on: same seed, same direction; different seed,
substantially different direction.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind
from visionforge.domain.selection import Candidate

PROJECT = ProjectId(uuid.UUID("11111111-1111-1111-1111-111111111111"))

#: Width of the test embeddings. Not 512: nothing reads the dimension, and a
#: sixteen-float vector is one a failing test can print.
VECTOR_DIM = 16


def vector(seed: int, dim: int = VECTOR_DIM) -> tuple[float, ...]:
    """A deterministic unit vector. Same seed, same direction, every run.

    Generated from a linear congruential sequence rather than from ``sin`` over
    a shared phase, because the latter makes consecutive seeds far more alike
    than two genuinely different clips would be -- seeds 3 and 4 came out at a
    cosine of 0.9, which the diversity gate correctly treats as the same shot.
    Fixtures that accidentally encode "everything is a duplicate" make the
    diversity tests pass for the wrong reason.
    """
    state = (seed + 1) * 2_654_435_761 % 2**32
    raw: list[float] = []
    for _ in range(dim):
        state = (state * 1_664_525 + 1_013_904_223) % 2**32
        raw.append(state / 2**31 - 1.0)
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return tuple(value / norm for value in raw)


def phash_for(index: int) -> str:
    """A 64-bit hash far from every other index's.

    Multiplied by a large odd constant so that consecutive indices are not
    within the pHash duplicate distance -- ``f"{index:016x}"`` differs by one
    bit and would have every clip suppressed as a near-duplicate before the
    editorial layer saw any of them.
    """
    return f"{(index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF:016x}"


def clip(
    index: int,
    *,
    motion: float | None = 0.2,
    motion_spread: float | None = 0.02,
    motion_peak_ms: int | None = None,
    saturation: float | None = 0.4,
    faces: tuple[bool, ...] = (),
    face_area: float | None = None,
    duration_ms: int = 8_000,
    blur_score: float = 300.0,
    contrast: float = 45.0,
    luminance: float = 128.0,
    subject: int | None = None,
    #: Set to drop the CLIP vector entirely, which is what an asset the GPU
    #: lane has not reached looks like.
    unembedded: bool = False,
    scene_boundaries_ms: tuple[int, ...] = (),
    has_audio: bool = True,
    ready: bool = True,
    **overrides: Any,
) -> Candidate:
    """One usable clip. Override whatever the test is actually about."""
    return Candidate(
        media_id=MediaId(uuid.uuid4()),
        kind=MediaKind.VIDEO,
        is_ready=ready,
        duration_ms=duration_ms,
        width=1280,
        height=720,
        blur_score=blur_score,
        contrast=contrast,
        mean_luminance=luminance,
        clipped_ratio=0.01,
        phash=phash_for(index + 1),
        scene_count=1 + len(scene_boundaries_ms),
        motion=motion,
        saturation=saturation,
        has_faces=bool(faces) and any(faces),
        has_audio=has_audio,
        sequence=index,
        motion_spread=motion_spread,
        face_area=face_area,
        face_frames=faces,
        scene_boundaries_ms=scene_boundaries_ms,
        mean_scene_ms=duration_ms // (1 + len(scene_boundaries_ms)),
        motion_peak_ms=motion_peak_ms if motion_peak_ms is not None else duration_ms // 2,
        embedding=(None if unembedded else vector(subject if subject is not None else index)),
        **overrides,
    )


def football_project() -> list[Candidate]:
    """Eleven clips with the shape a football highlight should be able to find.

    Built around the two cases this phase exists to handle, deliberately kept
    apart so a test can fail on one without the other:

    - **subject 4** is the peak, and it is the *worst* clip in the project on
      every technical measure -- shot on a long lens by somebody running. A
      ranker drops it; an editor does not, because it is the only record of the
      moment the edit is about.
    - **subject 5** is three near-identical angles on an ordinary moment, each
      sharper and better exposed than the peak. A ranker takes all three.
    """
    return [
        # Wide, still, nobody close: an establishing shot.
        clip(0, motion=0.02, duration_ms=9_000, subject=0),
        clip(1, motion=0.08, duration_ms=7_000, subject=1),
        # The build: rising movement.
        clip(2, motion=0.15, duration_ms=6_000, subject=2),
        clip(3, motion=0.24, duration_ms=6_000, subject=3),
        # The peak. Most motion in the project by a clear margin, and soft.
        clip(
            4,
            motion=0.34,
            motion_spread=0.10,
            motion_peak_ms=3_500,
            duration_ms=5_000,
            blur_score=95.0,
            contrast=20.0,
            subject=4,
        ),
        # Three angles on one ordinary moment. Technically excellent, and the
        # same shot three times.
        clip(5, motion=0.20, duration_ms=5_000, blur_score=560.0, subject=5),
        clip(6, motion=0.20, duration_ms=5_000, blur_score=555.0, subject=5),
        clip(7, motion=0.19, duration_ms=5_000, blur_score=550.0, subject=5),
        # The reaction: faces, close, calm.
        clip(8, motion=0.04, duration_ms=6_000, faces=(True,) * 5, face_area=0.14, subject=8),
        # Somebody walking out of frame.
        clip(
            9,
            motion=0.12,
            duration_ms=6_000,
            faces=(True, True, True, False, False),
            face_area=0.05,
            subject=9,
        ),
        # A calm, darkening last shot.
        clip(10, motion=0.03, duration_ms=8_000, luminance=70.0, subject=10),
    ]


def nature_project() -> list[Candidate]:
    """Twelve calm, varied, held shots. Nothing in it is an "action" clip."""
    return [
        clip(
            index,
            motion=0.02 + index * 0.012,
            duration_ms=9_000 + index * 400,
            saturation=0.35 + index * 0.02,
            subject=index,
        )
        for index in range(12)
    ]


def gaming_project() -> list[Candidate]:
    """Twelve busy screen-capture clips, sharp and saturated throughout."""
    return [
        clip(
            index,
            motion=0.18 + index * 0.014,
            motion_spread=0.03 + (index % 3) * 0.02,
            duration_ms=6_000,
            blur_score=520.0,
            saturation=0.6,
            subject=index,
        )
        for index in range(12)
    ]


__all__ = [
    "PROJECT",
    "VECTOR_DIM",
    "clip",
    "football_project",
    "gaming_project",
    "nature_project",
    "phash_for",
    "vector",
]
