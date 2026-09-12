"""The EditPlan: a declarative description of an edit.

This is the contract that boxes the planner in. A planner -- today a rules
engine, later an LLM behind the same interface -- may emit **only** this
structure: typed operations drawn from a closed vocabulary, referring to media
by id.

It cannot emit a shell command, an FFmpeg argument, a filesystem path, or
anything executable, because there is nowhere in the schema to put one. That is
a stronger guarantee than validating strings after the fact, and it is the whole
reason the plan exists as a separate artefact rather than the planner calling
the renderer directly.

    planner -> EditPlan -> validate -> Timeline -> RenderSpec -> argv[]
                             ^
                             nothing reaches FFmpeg without passing here

The schema is deliberately small. Phase 4 supports a source clip, a trim, an
order, a scale/crop fit, an output format and an audio choice. Transitions and
effects are declared as enums with a single member each, so adding one later is
an extension rather than a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from visionforge.domain.errors import PermanentError
from visionforge.domain.ids import MediaId, ProjectId

# --------------------------------------------------------------------- limits
#: Bounds every plan is checked against. These are product limits, not guesses:
#: they cap what a single render can cost on a 6-core laptop with 7.4 GB of RAM.
MIN_SEGMENT_MS = 300
MAX_SEGMENT_MS = 30_000
MAX_SEGMENTS = 40
MIN_OUTPUT_MS = 1_000
MAX_OUTPUT_MS = 10 * 60 * 1_000
MIN_DIMENSION = 16
MAX_DIMENSION = 3840
MIN_FPS = 1
MAX_FPS = 120


class AspectRatio(StrEnum):
    """Supported output shapes. A closed set, so no arbitrary geometry."""

    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"

    @property
    def ratio(self) -> float:
        width, height = (int(part) for part in self.value.split(":"))
        return width / height


class FitMode(StrEnum):
    """How a source frame is fitted into the output rectangle."""

    #: Scale to fill, then centre-crop the overflow. No bars, some content lost.
    COVER = "cover"
    #: Scale to fit, then pad. All content kept, bars added.
    CONTAIN = "contain"


class TransitionKind(StrEnum):
    """Phase 4 ships hard cuts only.

    Declared as an enum with one member rather than omitted, so that adding a
    dissolve later extends a closed vocabulary instead of introducing one.
    """

    CUT = "cut"


class AudioMode(StrEnum):
    NONE = "none"
    #: Keep the audio of the source clips, concatenated with the video.
    SOURCE = "source"


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """What the finished video should be."""

    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    width: int = 1280
    height: int = 720
    fps: int = 30
    fit: FitMode = FitMode.COVER
    audio: AudioMode = AudioMode.NONE


@dataclass(frozen=True, slots=True)
class Segment:
    """One clip's contribution to the edit.

    ``source_in_ms``/``source_out_ms`` are a trim *within the source*; position
    on the timeline is implied by ``order``, because Phase 4 has one video track
    and no gaps. Making position explicit would invite plans with overlaps and
    holes that the compiler would then have to reject.
    """

    media_id: MediaId
    order: int
    source_in_ms: int
    source_out_ms: int
    transition_in: TransitionKind = TransitionKind.CUT

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms


@dataclass(frozen=True, slots=True)
class EditPlan:
    """A complete, declarative edit.

    Frozen: a plan is a record of a decision. Re-planning produces a new one
    rather than mutating the old, which is what makes a render reproducible from
    the plan it was built from.
    """

    project_id: ProjectId
    segments: tuple[Segment, ...]
    output: OutputSpec = field(default_factory=OutputSpec)
    planner: str = "rules-engine"
    planner_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_media_ids(self) -> tuple[MediaId, ...]:
        return tuple(segment.media_id for segment in self.segments)

    @property
    def total_duration_ms(self) -> int:
        return sum(segment.duration_ms for segment in self.segments)

    @property
    def ordered_segments(self) -> tuple[Segment, ...]:
        return tuple(sorted(self.segments, key=lambda s: s.order))

    def as_payload(self) -> dict[str, Any]:
        """Serialise for JSONB storage and for the API."""
        return {
            "project_id": str(self.project_id),
            "planner": self.planner,
            "planner_version": self.planner_version,
            "output": {
                "aspect_ratio": self.output.aspect_ratio.value,
                "width": self.output.width,
                "height": self.output.height,
                "fps": self.output.fps,
                "fit": self.output.fit.value,
                "audio": self.output.audio.value,
            },
            "segments": [
                {
                    "media_id": str(segment.media_id),
                    "order": segment.order,
                    "source_in_ms": segment.source_in_ms,
                    "source_out_ms": segment.source_out_ms,
                    "duration_ms": segment.duration_ms,
                    "transition_in": segment.transition_in.value,
                }
                for segment in self.ordered_segments
            ],
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------ validation
@dataclass(frozen=True, slots=True)
class PlanViolation:
    """One reason a plan was rejected. Machine-readable, not just a message."""

    code: str
    message: str
    segment_order: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "segment_order": self.segment_order,
        }


class PlanInvalidError(PermanentError):
    """A plan failed validation. Carries every violation, not just the first.

    **Permanent, not transient.** A plan that references deleted media, or reaches
    into another project, will fail identically on every retry: the plan is fixed
    and so is the world it disagrees with. Classifying it as transient would burn
    the retry budget and delay the report by minutes for no chance of success --
    the same reasoning as a worker without FFmpeg.

    Fixing it means re-planning, which produces a *new* plan; this one never
    becomes valid.
    """

    code = "plan_invalid"
    http_status = 422

    def __init__(self, violations: list[PlanViolation]) -> None:
        self.violations = violations
        super().__init__(
            f"edit plan rejected ({len(violations)} violation(s)): "
            + "; ".join(v.message for v in violations),
            hint="Re-generate the plan; the media it references has changed.",
        )


@dataclass(frozen=True, slots=True)
class MediaFact:
    """What the validator is allowed to know about a referenced asset.

    Deliberately not the ORM row. The validator needs to check that media exists,
    belongs to the project, is renderable, and is long enough for the requested
    trim -- and nothing else. Passing the row would let a future check quietly
    start depending on a storage key.
    """

    media_id: MediaId
    project_id: ProjectId
    is_renderable: bool
    duration_ms: int | None
    width: int | None = None
    height: int | None = None


def validate_plan(plan: EditPlan, media_facts: dict[MediaId, MediaFact]) -> list[PlanViolation]:
    """Check a plan against structure, bounds and the media it claims to use.

    Returns every violation rather than raising on the first, so a planner (or a
    person) gets the whole list in one pass. ``assert_valid`` is the raising
    wrapper used at the gate.
    """
    violations: list[PlanViolation] = []
    out = plan.output

    # --- output ---
    if not MIN_DIMENSION <= out.width <= MAX_DIMENSION:
        violations.append(
            PlanViolation(
                "output_width", f"width {out.width} outside {MIN_DIMENSION}-{MAX_DIMENSION}"
            )
        )
    if not MIN_DIMENSION <= out.height <= MAX_DIMENSION:
        violations.append(
            PlanViolation(
                "output_height", f"height {out.height} outside {MIN_DIMENSION}-{MAX_DIMENSION}"
            )
        )
    if out.width % 2 or out.height % 2:
        # H.264 with yuv420p cannot encode odd dimensions. Catching it here
        # turns a cryptic encoder failure into a plan-level rejection.
        violations.append(
            PlanViolation(
                "output_dimensions_odd",
                f"{out.width}x{out.height} must both be even for H.264 4:2:0",
            )
        )
    if not MIN_FPS <= out.fps <= MAX_FPS:
        violations.append(PlanViolation("output_fps", f"fps {out.fps} outside {MIN_FPS}-{MAX_FPS}"))
    if out.width and out.height:
        actual = out.width / out.height
        if abs(actual - out.aspect_ratio.ratio) > 0.02:
            violations.append(
                PlanViolation(
                    "aspect_mismatch",
                    f"{out.width}x{out.height} is not {out.aspect_ratio.value}",
                )
            )

    # --- segments ---
    if not plan.segments:
        violations.append(PlanViolation("no_segments", "a plan must have at least one segment"))
    if len(plan.segments) > MAX_SEGMENTS:
        violations.append(
            PlanViolation(
                "too_many_segments", f"{len(plan.segments)} segments exceeds {MAX_SEGMENTS}"
            )
        )

    orders = [segment.order for segment in plan.segments]
    if len(set(orders)) != len(orders):
        violations.append(PlanViolation("duplicate_order", "segment orders must be unique"))

    for segment in plan.ordered_segments:
        violations.extend(_validate_segment(segment, plan, media_facts))

    # --- total duration ---
    total = plan.total_duration_ms
    if plan.segments and not MIN_OUTPUT_MS <= total <= MAX_OUTPUT_MS:
        violations.append(
            PlanViolation(
                "output_duration",
                f"total {total} ms outside {MIN_OUTPUT_MS}-{MAX_OUTPUT_MS} ms",
            )
        )

    return violations


def _validate_segment(
    segment: Segment, plan: EditPlan, media_facts: dict[MediaId, MediaFact]
) -> list[PlanViolation]:
    violations: list[PlanViolation] = []
    order = segment.order

    if segment.source_in_ms < 0:
        violations.append(
            PlanViolation("negative_in", f"source_in_ms {segment.source_in_ms} is negative", order)
        )
    if segment.source_out_ms <= segment.source_in_ms:
        violations.append(
            PlanViolation(
                "non_positive_duration",
                f"source_out_ms {segment.source_out_ms} must exceed "
                f"source_in_ms {segment.source_in_ms}",
                order,
            )
        )
        # Every further check on this segment depends on a sane range.
        return violations

    duration = segment.duration_ms
    if duration < MIN_SEGMENT_MS:
        violations.append(
            PlanViolation("segment_too_short", f"{duration} ms below {MIN_SEGMENT_MS} ms", order)
        )
    if duration > MAX_SEGMENT_MS:
        violations.append(
            PlanViolation("segment_too_long", f"{duration} ms above {MAX_SEGMENT_MS} ms", order)
        )

    fact = media_facts.get(segment.media_id)
    if fact is None:
        violations.append(
            PlanViolation("unknown_media", f"media {segment.media_id} does not exist", order)
        )
        return violations

    # The authorization check, enforced in the domain rather than only at the
    # API. A plan that references another project's media is invalid even if
    # something upstream failed to notice.
    if fact.project_id != plan.project_id:
        violations.append(
            PlanViolation(
                "cross_project_media",
                f"media {segment.media_id} belongs to another project",
                order,
            )
        )
    if not fact.is_renderable:
        violations.append(
            PlanViolation(
                "media_not_renderable",
                f"media {segment.media_id} has no usable video stream",
                order,
            )
        )
    if fact.duration_ms is not None and segment.source_out_ms > fact.duration_ms:
        violations.append(
            PlanViolation(
                "trim_past_end",
                f"source_out_ms {segment.source_out_ms} exceeds the source's "
                f"{fact.duration_ms} ms",
                order,
            )
        )
    return violations


def assert_valid(plan: EditPlan, media_facts: dict[MediaId, MediaFact]) -> None:
    """The gate. Nothing reaches the timeline compiler without passing it."""
    violations = validate_plan(plan, media_facts)
    if violations:
        raise PlanInvalidError(violations)


def media_id_from(value: str | UUID) -> MediaId:
    return MediaId(value if isinstance(value, UUID) else UUID(value))


def plan_from_payload(payload: dict[str, Any]) -> EditPlan:
    """Rebuild a typed plan from the JSON it was stored as.

    Lives in the domain because both the renderer and the API need it, and
    because reconstructing into the typed classes re-checks the document against
    the current schema: a payload written by an older version with a field that
    no longer exists fails here rather than producing a subtly wrong render.

    Raises ``KeyError``/``ValueError`` on a malformed document; callers treat
    that as a permanent failure, since a stored plan does not repair itself.
    """
    output = payload["output"]
    return EditPlan(
        project_id=ProjectId(UUID(str(payload["project_id"]))),
        segments=tuple(
            Segment(
                media_id=media_id_from(segment["media_id"]),
                order=int(segment["order"]),
                source_in_ms=int(segment["source_in_ms"]),
                source_out_ms=int(segment["source_out_ms"]),
                transition_in=TransitionKind(segment.get("transition_in", "cut")),
            )
            for segment in payload["segments"]
        ),
        output=OutputSpec(
            aspect_ratio=AspectRatio(output["aspect_ratio"]),
            width=int(output["width"]),
            height=int(output["height"]),
            fps=int(output["fps"]),
            fit=FitMode(output["fit"]),
            audio=AudioMode(output["audio"]),
        ),
        planner=str(payload.get("planner", "unknown")),
        planner_version=str(payload.get("planner_version", "0")),
        metadata=dict(payload.get("metadata", {})),
    )


__all__ = [
    "MAX_OUTPUT_MS",
    "MAX_SEGMENTS",
    "MAX_SEGMENT_MS",
    "MIN_OUTPUT_MS",
    "MIN_SEGMENT_MS",
    "AspectRatio",
    "AudioMode",
    "EditPlan",
    "FitMode",
    "MediaFact",
    "OutputSpec",
    "PlanInvalidError",
    "PlanViolation",
    "Segment",
    "TransitionKind",
    "assert_valid",
    "media_id_from",
    "plan_from_payload",
    "validate_plan",
]
