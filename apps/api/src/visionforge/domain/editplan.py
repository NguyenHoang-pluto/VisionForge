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

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from visionforge.domain.effects import (
    EFFECT_BOUNDS,
    MAX_EFFECTS_PER_SEGMENT,
    MIN_EFFECT_MS,
    Effect,
    output_duration_ms,
    speed_of,
)
from visionforge.domain.errors import PermanentError
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import MediaKind, MediaStatus
from visionforge.domain.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUE_MS,
    MAX_CUES,
    MIN_CUE_GAP_MS,
    MIN_CUE_MS,
    SubtitleTrack,
)

# --------------------------------------------------------------------- limits
#: Bounds every plan is checked against. These are product limits, not guesses:
#: they cap what a single render can cost on a 6-core laptop with 7.4 GB of RAM.
MIN_SEGMENT_MS = 300
MAX_SEGMENT_MS = 30_000
MAX_SEGMENTS = 40

#: Transition bounds (Phase 9).
#:
#: The floor is two frames at 60 fps: below that the effect is a cut with extra
#: encoding. The ceiling is a judgement -- a four-second dissolve is a stylistic
#: choice, a forty-second one is a mistake that also costs a great deal of
#: encode time.
MIN_TRANSITION_MS = 80
MAX_TRANSITION_MS = 4_000

#: A crossfade eats into both neighbours. Allowing it to consume more than this
#: share of either would leave a "clip" that is entirely dissolve -- visible as a
#: smear rather than as an edit, and impossible for ``xfade`` to place.
MAX_TRANSITION_SHARE = 0.5
MIN_OUTPUT_MS = 1_000
MAX_OUTPUT_MS = 10 * 60 * 1_000
MIN_DIMENSION = 16
MAX_DIMENSION = 3840
MIN_FPS = 1
MAX_FPS = 120

# --- audio (Phase 7) ---
#: Shortest usable music cue. Below half a second a "bed" is a click.
MIN_MUSIC_MS = 500
#: A cue may be as long as the longest renderable output; anything past the end
#: of the video is trimmed by the compiler rather than rejected here.
MAX_MUSIC_MS = MAX_OUTPUT_MS
#: Linear gain, where 1.0 is unity. The ceiling is 2.0 rather than unbounded
#: because there is no limiter in the graph: a plan that asks for 8x would
#: simply clip, and a plan that cannot be rendered cleanly is not a plan.
MIN_GAIN = 0.0
MAX_GAIN = 2.0
#: A fade longer than this is a structural choice the cue vocabulary does not
#: express; it is also longer than most of the edits this system produces.
MAX_FADE_MS = 30_000


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
    """How one segment gives way to the next. A closed vocabulary.

    Phase 4 shipped ``CUT`` alone and said a dissolve would extend this rather
    than introduce a vocabulary; Phase 9 is that extension. Four members, and
    the restraint is deliberate: a wipe, an iris and a page curl are each a
    filter, a parameter set and a timing rule, and shipping twelve of them
    badly is worse than shipping four that are exact.

    They divide into two kinds, and the division is the whole of the timing
    model:

    **Overlapping.** ``CROSSFADE`` plays the tail of one clip and the head of
    the next at the same time, so the programme is *shorter* than the sum of its
    segments by exactly the overlap.

    **Overlaid.** ``FADE_IN`` and ``FADE_TO_BLACK`` are drawn on top of a clip
    that plays its full length. They consume no time and move nothing.
    """

    CUT = "cut"
    #: Dissolve from the previous segment into this one. Consumes time.
    CROSSFADE = "crossfade"
    #: This segment fades up from black. Consumes no time.
    FADE_IN = "fade_in"
    #: This segment fades down to black at its end. Consumes no time.
    FADE_TO_BLACK = "fade_to_black"

    @property
    def consumes_time(self) -> bool:
        """Whether this transition shortens the programme.

        The single question the timeline arithmetic asks. Written as a property
        of the kind rather than as a set the callers each keep, because a fifth
        member added without answering it would otherwise silently default to
        "no" and produce a plan whose reported duration is wrong.
        """
        return self is TransitionKind.CROSSFADE

    @property
    def needs_previous(self) -> bool:
        """Whether this transition is meaningless on the first segment."""
        return self is TransitionKind.CROSSFADE


class QualityPreset(StrEnum):
    """Encoder effort, named for the outcome rather than the setting.

    Defined here rather than alongside the style profiles because ``OutputSpec``
    needs it and ``style`` already imports this module; putting it there would
    make the dependency circular. The CRF and preset each level maps to live in
    ``domain.style``, which is where the numbers behind a name belong.
    """

    DRAFT = "draft"
    BALANCED = "balanced"
    HIGH = "high"


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

    #: Encoder effort, as a name rather than a CRF. The plan says how good the
    #: output should be; only the render spec knows what that costs in encoder
    #: settings. ``BALANCED`` is the Phase 4 behaviour exactly, so a plan written
    #: before this field existed re-renders to the same bytes.
    quality: QualityPreset = QualityPreset.BALANCED

    #: Linear gain applied to the clips' own audio, independent of any music.
    #: Separate from ``audio`` because "keep the source audio" and "how loud"
    #: are different decisions: ducking dialogue under a music bed is the
    #: common case and it must not require turning the source off.
    #: ``1.0`` is unity, so a plan written before this field existed sounds
    #: identical.
    source_gain: float = 1.0


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
    #: How long the incoming transition runs. Ignored for ``CUT``, which has no
    #: duration by definition; validated against both neighbours for the kinds
    #: that do.
    transition_ms: int = 0
    #: Effects applied to this segment (Phase 9). A tuple, because a clip may
    #: reasonably carry a zoom *and* a colour adjustment, and ordering them is
    #: the compiler's job rather than the caller's.
    effects: tuple[Effect, ...] = ()

    @property
    def duration_ms(self) -> int:
        """How long this segment's source contributes.

        Note what this is *not*: the time it occupies on the timeline. A segment
        entered by a crossfade overlaps its predecessor, so the programme grows
        by less than this. ``Timeline`` owns that arithmetic; a segment only
        knows its own trim.
        """
        return self.source_out_ms - self.source_in_ms

    @property
    def overlap_ms(self) -> int:
        """How much of this segment plays over the previous one."""
        return self.transition_ms if self.transition_in.consumes_time else 0

    @property
    def output_duration_ms(self) -> int:
        """How long this segment occupies once its speed effects are applied.

        Half speed doubles a clip. This is the number the timeline lays out and
        the number the renderer produces; ``duration_ms`` is the trim it was cut
        from, and the two are equal only when nothing changed the rate.
        """
        return output_duration_ms(self.duration_ms, self.effects)

    @property
    def speed(self) -> float:
        return speed_of(self.effects)


@dataclass(frozen=True, slots=True)
class MusicCue:
    """One piece of music laid under the edit.

    A closed vocabulary of seven numbers and an id. There is no filter field, no
    filename and no place to put an FFmpeg argument -- the same structural
    guarantee ``Segment`` gives the video track, applied to audio, because audio
    filters are exactly as capable of running a command as video ones.

    One cue, not a list. A montage with two beds and a crossfade between them is
    a real edit, but it is a *different* edit: it needs overlap rules, relative
    ordering and a mix policy, all of which the video track deliberately does not
    have either. Phase 7 ships the case that covers a highlight reel, and leaves
    the vocabulary extendable rather than pre-emptively general.

    ``timeline_start_ms`` is where the cue begins in the *output*, and
    ``source_in_ms``/``source_out_ms`` are a trim within the *music file*. The
    two coordinate systems are kept apart for the same reason ``TimelineClip``
    keeps them apart: conflating them is how audio drifts against picture.
    """

    media_id: MediaId
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int = 0
    #: Linear, 1.0 = unity. Exposed to the user as a percentage.
    gain: float = 1.0
    fade_in_ms: int = 0
    fade_out_ms: int = 0

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms

    @property
    def timeline_end_ms(self) -> int:
        return self.timeline_start_ms + self.duration_ms

    def as_payload(self) -> dict[str, Any]:
        return {
            "media_id": str(self.media_id),
            "source_in_ms": self.source_in_ms,
            "source_out_ms": self.source_out_ms,
            "timeline_start_ms": self.timeline_start_ms,
            "duration_ms": self.duration_ms,
            "gain": round(self.gain, 4),
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
        }


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
    #: The music bed, if the edit has one. ``None`` is the Phase 4-6 plan
    #: exactly -- which is why this is an optional field on the plan rather than
    #: a new member of ``AudioMode``: a plan written before Phase 7 deserialises
    #: to ``None`` and renders to the same bytes it always did.
    music: MusicCue | None = None
    #: Subtitles, if the edit has any (Phase 9). Optional for the same reason
    #: ``music`` is: a plan written before Phase 9 deserialises to ``None`` and
    #: renders to the same bytes.
    subtitles: SubtitleTrack | None = None
    planner: str = "rules-engine"
    planner_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_media_ids(self) -> tuple[MediaId, ...]:
        """The video sources, in segment order. Unchanged by Phase 7."""
        return tuple(segment.media_id for segment in self.segments)

    @property
    def referenced_media_ids(self) -> tuple[MediaId, ...]:
        """Every asset this plan names, deduplicated, order preserved.

        What the render worker must resolve and download -- which is the video
        sources *and* the music bed. Separate from ``source_media_ids`` because
        that one means "the clips", and several callers reasonably want only
        those; a worker fetching bytes wants all of them, and the difference is
        exactly the bug where the music file is validated and then never
        downloaded.
        """
        ids = list(dict.fromkeys(self.source_media_ids))
        if self.music is not None and self.music.media_id not in ids:
            ids.append(self.music.media_id)
        return tuple(ids)

    @property
    def total_duration_ms(self) -> int:
        """How long the programme runs.

        Three things make this more than a sum, and getting any of them wrong
        produces a plan that reports one length and renders another:

        - a speed effect changes how long a segment *plays* (half speed doubles
          it), so each segment contributes its output duration, not its trim;
        - a crossfade overlaps two segments, so the programme is shorter than
          their sum by exactly the overlap;
        - a fade to or from black is drawn on top of a clip that plays its full
          length and changes nothing.

        The arithmetic lives here, and ``compile_timeline`` lays clips out using
        the same two helpers, so the plan and the timeline cannot disagree.
        """
        segments = self.ordered_segments
        if not segments:
            return 0
        total = sum(segment.output_duration_ms for segment in segments)
        # The first segment has nothing to overlap with, whatever it claims.
        total -= sum(segment.overlap_ms for segment in segments[1:])
        return total

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
                "quality": self.output.quality.value,
                "source_gain": round(self.output.source_gain, 4),
            },
            "music": self.music.as_payload() if self.music else None,
            "subtitles": self.subtitles.as_payload() if self.subtitles else None,
            "segments": [
                {
                    "media_id": str(segment.media_id),
                    "order": segment.order,
                    "source_in_ms": segment.source_in_ms,
                    "source_out_ms": segment.source_out_ms,
                    "duration_ms": segment.duration_ms,
                    "transition_in": segment.transition_in.value,
                    "transition_ms": segment.transition_ms,
                    "effects": [effect.as_payload() for effect in segment.effects],
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
    #: True when the violation is about the music cue rather than a segment, so
    #: an editor can point at the audio lane instead of guessing.
    is_music: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "segment_order": self.segment_order,
            "is_music": self.is_music,
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
    #: Ready, and an audio asset in its own right. The one fact a music cue
    #: needs, and deliberately not "has an audio stream": Phase 7 beds are
    #: project-owned audio files, so pointing a cue at the soundtrack of a video
    #: is a different feature and is refused here rather than half-working.
    is_audio_asset: bool = False

    @classmethod
    def from_media(
        cls,
        *,
        media_id: MediaId,
        project_id: ProjectId,
        kind: MediaKind,
        status: MediaStatus,
        duration_ms: int | None,
        width: int | None = None,
        height: int | None = None,
    ) -> MediaFact:
        """The one place an asset's kind and status become planning facts.

        Both the planner's service and the render worker need this mapping, and
        before Phase 7 both wrote it out inline. That was survivable with one
        derived flag; with two it is a guarantee that they will disagree the
        first time a third is added -- and the two callers sit on opposite sides
        of the validation gate, so a disagreement would mean a plan that
        validates at creation and fails at render.

        Note what "usable" means on each axis. A video is renderable when it is
        ready; an audio file is a music source when it is ready. A video with an
        audio stream is neither a music source nor an error -- it is simply not
        what a cue may point at.
        """
        ready = status is MediaStatus.READY
        return cls(
            media_id=media_id,
            project_id=project_id,
            is_renderable=ready and kind is MediaKind.VIDEO,
            duration_ms=duration_ms,
            width=width,
            height=height,
            is_audio_asset=ready and kind is MediaKind.AUDIO,
        )


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

    violations.extend(_validate_transitions(plan))
    violations.extend(_validate_effects(plan))
    violations.extend(_validate_subtitles(plan))

    # --- total duration ---
    total = plan.total_duration_ms
    if plan.segments and not MIN_OUTPUT_MS <= total <= MAX_OUTPUT_MS:
        violations.append(
            PlanViolation(
                "output_duration",
                f"total {total} ms outside {MIN_OUTPUT_MS}-{MAX_OUTPUT_MS} ms",
            )
        )

    # --- audio ---
    if not MIN_GAIN <= out.source_gain <= MAX_GAIN:
        violations.append(
            PlanViolation(
                "source_gain_range",
                f"source gain {out.source_gain} outside {MIN_GAIN}-{MAX_GAIN}",
            )
        )
    if plan.music is not None:
        violations.extend(_validate_music(plan.music, plan, media_facts, total))

    return violations


def _validate_music(
    cue: MusicCue,
    plan: EditPlan,
    media_facts: dict[MediaId, MediaFact],
    timeline_ms: int,
) -> list[PlanViolation]:
    """Check the music cue against its own bounds and the asset it names.

    Every check mirrors one the video track already has, for the same reason it
    has it: a trim past the end of a source, a range that is not a range, or an
    asset belonging to somebody else. The audio-specific ones are gain, fades
    and whether the cue plays at all -- a bed that starts after the video ends
    is silence the user asked for by mistake, and saying so is more useful than
    rendering it.
    """
    violations: list[PlanViolation] = []

    if cue.source_in_ms < 0:
        violations.append(
            PlanViolation(
                "music_negative_in",
                f"music source_in_ms {cue.source_in_ms} is negative",
                is_music=True,
            )
        )
    if cue.source_out_ms <= cue.source_in_ms:
        violations.append(
            PlanViolation(
                "music_non_positive_duration",
                f"music source_out_ms {cue.source_out_ms} must exceed "
                f"source_in_ms {cue.source_in_ms}",
                is_music=True,
            )
        )
        # Every remaining check needs a sane range.
        return violations

    duration = cue.duration_ms
    if duration < MIN_MUSIC_MS:
        violations.append(
            PlanViolation(
                "music_too_short",
                f"music cue {duration} ms below {MIN_MUSIC_MS} ms",
                is_music=True,
            )
        )
    if duration > MAX_MUSIC_MS:
        violations.append(
            PlanViolation(
                "music_too_long",
                f"music cue {duration} ms above {MAX_MUSIC_MS} ms",
                is_music=True,
            )
        )

    if cue.timeline_start_ms < 0:
        violations.append(
            PlanViolation(
                "music_negative_start",
                f"music timeline_start_ms {cue.timeline_start_ms} is negative",
                is_music=True,
            )
        )
    elif timeline_ms and cue.timeline_start_ms >= timeline_ms:
        violations.append(
            PlanViolation(
                "music_starts_after_end",
                f"music starts at {cue.timeline_start_ms} ms, after the "
                f"{timeline_ms} ms timeline ends",
                is_music=True,
            )
        )

    if not MIN_GAIN <= cue.gain <= MAX_GAIN:
        violations.append(
            PlanViolation(
                "music_gain_range",
                f"music gain {cue.gain} outside {MIN_GAIN}-{MAX_GAIN}",
                is_music=True,
            )
        )

    for name, value in (("fade_in_ms", cue.fade_in_ms), ("fade_out_ms", cue.fade_out_ms)):
        if value < 0:
            violations.append(
                PlanViolation(
                    "music_negative_fade", f"music {name} {value} is negative", is_music=True
                )
            )
        elif value > MAX_FADE_MS:
            violations.append(
                PlanViolation(
                    "music_fade_too_long",
                    f"music {name} {value} ms above {MAX_FADE_MS} ms",
                    is_music=True,
                )
            )

    # Overlapping fades would ask afade to ramp up and down over the same
    # samples, which FFmpeg resolves by silently preferring one of them.
    if cue.fade_in_ms >= 0 and cue.fade_out_ms >= 0 and cue.fade_in_ms + cue.fade_out_ms > duration:
        violations.append(
            PlanViolation(
                "music_fades_overlap",
                f"fades total {cue.fade_in_ms + cue.fade_out_ms} ms, longer than "
                f"the {duration} ms cue",
                is_music=True,
            )
        )

    fact = media_facts.get(cue.media_id)
    if fact is None:
        violations.append(
            PlanViolation(
                "music_unknown_media",
                f"music asset {cue.media_id} does not exist",
                is_music=True,
            )
        )
        return violations

    if fact.project_id != plan.project_id:
        violations.append(
            PlanViolation(
                "music_cross_project_media",
                f"music asset {cue.media_id} belongs to another project",
                is_music=True,
            )
        )
    if not fact.is_audio_asset:
        violations.append(
            PlanViolation(
                "music_not_audio",
                f"media {cue.media_id} is not a ready audio asset",
                is_music=True,
            )
        )
    if fact.duration_ms is not None and cue.source_out_ms > fact.duration_ms:
        violations.append(
            PlanViolation(
                "music_trim_past_end",
                f"music source_out_ms {cue.source_out_ms} exceeds the track's "
                f"{fact.duration_ms} ms",
                is_music=True,
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


def _validate_transitions(plan: EditPlan) -> list[PlanViolation]:
    """Transitions, and the timing they are only allowed to imply.

    A crossfade is the only kind that consumes time, and it is the only one with
    anything to get wrong: it eats into both neighbours, so it has to be shorter
    than either can spare, and it needs a previous clip to dissolve from.
    """
    violations: list[PlanViolation] = []
    segments = plan.ordered_segments

    for index, segment in enumerate(segments):
        kind = segment.transition_in
        order = segment.order

        if kind is TransitionKind.CUT:
            # A cut has no duration by definition. A plan that sends one is
            # describing something it does not mean, and silently ignoring it
            # would let an editor show a slider that does nothing.
            if segment.transition_ms:
                violations.append(
                    PlanViolation(
                        "transition_duration_on_cut",
                        f"a cut cannot last {segment.transition_ms} ms",
                        order,
                    )
                )
            continue

        if not MIN_TRANSITION_MS <= segment.transition_ms <= MAX_TRANSITION_MS:
            violations.append(
                PlanViolation(
                    "transition_duration",
                    f"{segment.transition_ms} ms outside "
                    f"{MIN_TRANSITION_MS}-{MAX_TRANSITION_MS} ms",
                    order,
                )
            )
            continue

        if kind.needs_previous and index == 0:
            violations.append(
                PlanViolation(
                    "transition_without_previous",
                    f"{kind.value} needs a segment to come from",
                    order,
                )
            )
            continue

        if not kind.consumes_time:
            continue

        # A dissolve borrows from both clips. Either being too short to lend is
        # the same failure, and both are reported against the segment carrying
        # the transition, because that is the one the user chose.
        previous = segments[index - 1]
        share = int(
            min(previous.output_duration_ms, segment.output_duration_ms) * MAX_TRANSITION_SHARE
        )
        if segment.transition_ms > share:
            violations.append(
                PlanViolation(
                    "transition_too_long_for_neighbours",
                    f"{segment.transition_ms} ms exceeds {share} ms, "
                    f"half the shorter of the two clips",
                    order,
                )
            )

    return violations


def _validate_effects(plan: EditPlan) -> list[PlanViolation]:
    """Effect kinds, parameter ranges, windows, and the combinations that lie."""
    violations: list[PlanViolation] = []

    for segment in plan.ordered_segments:
        order = segment.order

        if len(segment.effects) > MAX_EFFECTS_PER_SEGMENT:
            violations.append(
                PlanViolation(
                    "too_many_effects",
                    f"{len(segment.effects)} effects exceeds {MAX_EFFECTS_PER_SEGMENT}",
                    order,
                )
            )

        kinds = [effect.kind for effect in segment.effects]
        if len(set(kinds)) != len(kinds):
            violations.append(
                PlanViolation(
                    "duplicate_effect",
                    "a segment cannot carry the same effect twice",
                    order,
                )
            )

        speeds = [kind for kind in kinds if kind.changes_duration]
        if len(speeds) > 1:
            # Two rates multiply to a third that nobody asked for.
            violations.append(
                PlanViolation(
                    "conflicting_speed",
                    "a segment can have one speed change, not two",
                    order,
                )
            )

        for effect in segment.effects:
            low, high, _ = EFFECT_BOUNDS[effect.kind]
            if not low <= effect.amount <= high:
                violations.append(
                    PlanViolation(
                        "effect_amount",
                        f"{effect.kind.value} {effect.amount} outside {low}-{high}",
                        order,
                    )
                )

            if effect.is_ranged and effect.kind.spans_whole_segment:
                violations.append(
                    PlanViolation(
                        "effect_cannot_be_ranged",
                        f"{effect.kind.value} applies to a whole clip, not part of one",
                        order,
                    )
                )
                continue

            if not effect.is_ranged:
                continue

            start, end = effect.window(segment.duration_ms)
            if start >= end:
                violations.append(
                    PlanViolation(
                        "effect_window_inverted",
                        f"{effect.kind.value} window {start}-{end} ms is empty",
                        order,
                    )
                )
            elif end - start < MIN_EFFECT_MS:
                violations.append(
                    PlanViolation(
                        "effect_window_too_short",
                        f"{end - start} ms below {MIN_EFFECT_MS} ms",
                        order,
                    )
                )
            if effect.end_ms is not None and effect.end_ms > segment.duration_ms:
                violations.append(
                    PlanViolation(
                        "effect_window_past_end",
                        f"{effect.end_ms} ms is past the clip's {segment.duration_ms} ms",
                        order,
                    )
                )

    return violations


def _validate_subtitles(plan: EditPlan) -> list[PlanViolation]:
    """Cue bounds, ordering, overlap and text length.

    Cues are in *timeline* coordinates, so they are checked against the
    programme's own length -- which already accounts for crossfade overlap and
    speed changes. A cue past the end would simply never be drawn, which is a
    silent failure and the kind this gate exists to convert into a loud one.
    """
    track = plan.subtitles
    if track is None:
        return []

    violations: list[PlanViolation] = []
    cues = track.ordered

    if len(cues) > MAX_CUES:
        violations.append(PlanViolation("too_many_cues", f"{len(cues)} cues exceeds {MAX_CUES}"))

    total = plan.total_duration_ms
    previous_end: int | None = None

    for index, cue in enumerate(cues):
        where = f"cue {index + 1}"

        if cue.start_ms < 0:
            violations.append(PlanViolation("cue_negative_start", f"{where} starts before zero"))
        if cue.end_ms <= cue.start_ms:
            violations.append(PlanViolation("cue_inverted", f"{where} ends at or before it starts"))
            continue

        if not MIN_CUE_MS <= cue.duration_ms <= MAX_CUE_MS:
            violations.append(
                PlanViolation(
                    "cue_duration",
                    f"{where} is {cue.duration_ms} ms, outside {MIN_CUE_MS}-{MAX_CUE_MS} ms",
                )
            )

        if not cue.text.strip():
            violations.append(PlanViolation("cue_empty", f"{where} has no text"))
        elif len(cue.text) > MAX_CUE_CHARS:
            violations.append(
                PlanViolation(
                    "cue_too_long",
                    f"{where} is {len(cue.text)} characters, over {MAX_CUE_CHARS}",
                )
            )

        if total and cue.end_ms > total:
            violations.append(
                PlanViolation(
                    "cue_past_end",
                    f"{where} ends at {cue.end_ms} ms, past the edit's {total} ms",
                )
            )

        if previous_end is not None and cue.start_ms < previous_end + MIN_CUE_GAP_MS:
            # Overlap is rejected rather than layered. Two cues on screen at
            # once is a different feature -- it needs a second row, a stacking
            # rule and a safe area that accounts for both -- and rendering them
            # on top of each other is not that feature.
            violations.append(
                PlanViolation(
                    "cue_overlap",
                    f"{where} starts at {cue.start_ms} ms, before {previous_end} ms",
                )
            )
        previous_end = max(previous_end or 0, cue.end_ms)

    return violations


def assert_valid(plan: EditPlan, media_facts: dict[MediaId, MediaFact]) -> None:
    """The gate. Nothing reaches the timeline compiler without passing it."""
    violations = validate_plan(plan, media_facts)
    if violations:
        raise PlanInvalidError(violations)


def media_id_from(value: str | UUID) -> MediaId:
    return MediaId(value if isinstance(value, UUID) else UUID(value))


# ---------------------------------------------------------------- hand-cutting
@dataclass(frozen=True, slots=True)
class Cut:
    """One clip as an editor placed it on the timeline.

    Deliberately has no ``order`` field. A hand-cut edit arrives as a sequence,
    and position on the timeline *is* the position in that sequence -- so the
    one way to express an overlap or a hole (two segments claiming the same
    order, or a gap between them) does not exist in the input. ``plan_from_cuts``
    is what turns a sequence into ordered segments, and it is the only way a
    manual plan gets built.
    """

    media_id: MediaId
    source_in_ms: int
    source_out_ms: int
    transition_in: TransitionKind = TransitionKind.CUT
    transition_ms: int = 0
    effects: tuple[Effect, ...] = ()


def plan_from_cuts(
    *,
    project_id: ProjectId,
    cuts: Sequence[Cut],
    output: OutputSpec,
    music: MusicCue | None = None,
    subtitles: SubtitleTrack | None = None,
    planner: str = "manual",
    planner_version: str = "1",
    metadata: dict[str, Any] | None = None,
) -> EditPlan:
    """Build a plan from an ordered sequence of cuts.

    The result is an ordinary ``EditPlan`` and goes through ``validate_plan``
    like any other. A timeline the user assembled by hand gets no weaker a gate
    than one a model proposed: the bounds on segment length, total duration and
    trim-past-end are the renderer's limits, and they do not care who chose the
    numbers.
    """
    return EditPlan(
        project_id=project_id,
        segments=tuple(
            Segment(
                media_id=cut.media_id,
                order=index,
                source_in_ms=cut.source_in_ms,
                source_out_ms=cut.source_out_ms,
                transition_in=cut.transition_in,
                transition_ms=cut.transition_ms,
                effects=cut.effects,
            )
            for index, cut in enumerate(cuts)
        ),
        output=output,
        music=music,
        subtitles=subtitles,
        planner=planner,
        planner_version=planner_version,
        metadata=dict(metadata or {}),
    )


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
                transition_ms=int(segment.get("transition_ms", 0)),
                effects=tuple(Effect.from_payload(effect) for effect in segment.get("effects", [])),
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
            # Absent in plans written before Phase 5. Defaulting rather than
            # failing is correct here: the default *is* what those plans were
            # rendered with.
            quality=QualityPreset(output.get("quality", QualityPreset.BALANCED.value)),
            # Absent before Phase 7. Unity is what those plans were rendered
            # with, so defaulting reproduces them exactly.
            source_gain=float(output.get("source_gain", 1.0)),
        ),
        music=_music_from_payload(payload.get("music")),
        subtitles=_subtitles_from_payload(payload.get("subtitles")),
        planner=str(payload.get("planner", "unknown")),
        planner_version=str(payload.get("planner_version", "0")),
        metadata=dict(payload.get("metadata", {})),
    )


def _subtitles_from_payload(raw: Any) -> SubtitleTrack | None:
    """Rebuild a subtitle track, or ``None`` for a plan written before Phase 9.

    An unknown style or position raises rather than falling back to a default:
    a stored plan naming a preset this build does not have is a plan that would
    render differently from the one that was approved, and silently substituting
    "clean" for it is the wrong kind of resilience.
    """
    if not isinstance(raw, dict):
        return None
    return SubtitleTrack.from_payload(raw)


def _music_from_payload(raw: Any) -> MusicCue | None:
    """Rebuild a cue from stored JSON. ``None`` and absent both mean no music."""
    if not raw:
        return None
    return MusicCue(
        media_id=media_id_from(raw["media_id"]),
        source_in_ms=int(raw["source_in_ms"]),
        source_out_ms=int(raw["source_out_ms"]),
        timeline_start_ms=int(raw.get("timeline_start_ms", 0)),
        gain=float(raw.get("gain", 1.0)),
        fade_in_ms=int(raw.get("fade_in_ms", 0)),
        fade_out_ms=int(raw.get("fade_out_ms", 0)),
    )


__all__ = [
    "MAX_FADE_MS",
    "MAX_GAIN",
    "MAX_MUSIC_MS",
    "MAX_OUTPUT_MS",
    "MAX_SEGMENTS",
    "MAX_SEGMENT_MS",
    "MIN_GAIN",
    "MIN_MUSIC_MS",
    "MIN_OUTPUT_MS",
    "MIN_SEGMENT_MS",
    "AspectRatio",
    "AudioMode",
    "Cut",
    "EditPlan",
    "FitMode",
    "MediaFact",
    "MusicCue",
    "OutputSpec",
    "PlanInvalidError",
    "PlanViolation",
    "QualityPreset",
    "Segment",
    "TransitionKind",
    "assert_valid",
    "media_id_from",
    "plan_from_cuts",
    "plan_from_payload",
    "validate_plan",
]
