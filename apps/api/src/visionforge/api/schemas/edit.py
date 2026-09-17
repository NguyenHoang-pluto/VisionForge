"""Request and response models for planning and rendering."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visionforge.domain.editbrief import MAX_REQUEST_CHARS
from visionforge.domain.editdelta import MAX_OPERATIONS
from visionforge.domain.editplan import (
    MAX_FADE_MS,
    MAX_GAIN,
    MAX_MUSIC_MS,
    MAX_OUTPUT_MS,
    MAX_SEGMENTS,
    MAX_TRANSITION_MS,
    MIN_GAIN,
    MIN_MUSIC_MS,
    MIN_OUTPUT_MS,
    AspectRatio,
    AudioMode,
    FitMode,
    QualityPreset,
    TransitionKind,
)
from visionforge.domain.effects import EFFECT_BOUNDS, MAX_EFFECTS_PER_SEGMENT, Effect, EffectKind
from visionforge.domain.llm_planner import PlannerEngine, PlannerMode
from visionforge.domain.planner import ClipOrder
from visionforge.domain.policy import StyleStrength
from visionforge.domain.story import PolicyId
from visionforge.domain.style import FPS_PRESETS, EditStyle
from visionforge.domain.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUES,
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
    clean_text,
)
from visionforge.domain.variants import VariantId


class MusicRequest(BaseModel):
    """A music bed, as *intent*.

    Note what is absent, and note that it is the same list absent from every
    other request model in this API: no path, no storage key, no codec, no
    filter string, no FFmpeg argument. A cue is a media id the caller already
    owns plus six numbers. There is nowhere in this schema to put a token, which
    is a stronger guarantee than sanitising one after the fact -- and it has to
    be, because an audio filter graph runs commands exactly as readily as a
    video one.

    The bounds here are the *client-facing* ones. They deliberately repeat what
    ``domain.editplan`` enforces rather than replacing it: this layer produces a
    422 with a field path, which is a better error, but the plan validator still
    runs against the real media rows and remains the authority. A value that
    slipped past Pydantic would still be refused before anything is stored.
    """

    media_id: UUID
    source_in_ms: int = Field(default=0, ge=0, le=MAX_MUSIC_MS)
    source_out_ms: int = Field(gt=0, le=MAX_MUSIC_MS)
    #: Where the bed starts in the *output*, not in the track.
    timeline_start_ms: int = Field(default=0, ge=0, le=MAX_OUTPUT_MS)
    #: Linear gain; 1.0 is unity. The UI shows it as a percentage.
    volume: float = Field(default=0.7, ge=MIN_GAIN, le=MAX_GAIN)
    fade_in_ms: int = Field(default=0, ge=0, le=MAX_FADE_MS)
    fade_out_ms: int = Field(default=1_500, ge=0, le=MAX_FADE_MS)

    @model_validator(mode="after")
    def _range_is_a_range(self) -> MusicRequest:
        """Cheap structural checks, so the common mistakes get a field error.

        Everything that needs the *database* -- does this track exist, is it
        audio, is it yours, is it long enough -- stays in the plan validator
        where the media rows are. This only rejects what is wrong on its face.
        """
        if self.source_out_ms <= self.source_in_ms:
            raise ValueError("source_out_ms must be greater than source_in_ms")
        length = self.source_out_ms - self.source_in_ms
        if length < MIN_MUSIC_MS:
            raise ValueError(f"the cue is {length} ms; the minimum is {MIN_MUSIC_MS} ms")
        if self.fade_in_ms + self.fade_out_ms > length:
            raise ValueError("fade_in_ms and fade_out_ms together exceed the cue")
        return self


class PlanCreateRequest(BaseModel):
    """Preferences for an automatic edit.

    Note what is absent: no width, no height, no codec, no CRF, no path. The
    caller chooses a shape, a length and a quality *level*; the server chooses
    the geometry and the encoder settings. There is no field here through which
    a client -- or a model reading a field a client set -- could influence what
    FFmpeg is handed.

    Several fields are ``None`` by default rather than carrying a literal. That
    is how "the user did not say" stays distinguishable from "the user chose the
    default", which is what lets a style supply its own defaults without
    overriding an explicit choice.
    """

    # --- how to plan ---
    mode: PlannerMode = PlannerMode.AUTOMATIC
    #: Which deterministic engine turns the request into clips. Defaults to the
    #: Phase 11 editorial engine, which is what automatic editing now means.
    #: ``"rules"`` selects the Phase 4 even split with Phase 5's directive
    #: planner in front of it -- kept reachable so the two generations can be
    #: run against the same footage, which is the only way to show that the new
    #: engine is not the old one carrying extra metadata.
    engine: PlannerEngine = PlannerEngine.EDITORIAL
    style: EditStyle | None = None
    #: What the user wants, in their own words. Bounded, and the only free text
    #: in the API. It reaches a model as data inside a delimited block, and it
    #: is never stored -- only a digest of it is.
    request_text: str | None = Field(default=None, max_length=MAX_REQUEST_CHARS)

    # --- what to produce ---
    target_duration_ms: int | None = Field(
        default=None, ge=MIN_OUTPUT_MS, le=MAX_OUTPUT_MS, examples=[25_000]
    )
    max_clips: int = Field(default=8, ge=1, le=40)
    min_clips: int = Field(default=2, ge=1, le=40)
    aspect_ratio: AspectRatio | None = None
    fps: int = Field(default=30, ge=1, le=60)
    fit: FitMode = FitMode.COVER
    audio: AudioMode | None = None
    order: ClipOrder | None = None
    quality: QualityPreset = QualityPreset.BALANCED
    #: Gain on the clips' own audio, independent of the bed. Ducking dialogue
    #: under music is the common case and must not require muting it.
    source_gain: float = Field(default=1.0, ge=MIN_GAIN, le=MAX_GAIN)

    # --- reference style (Phase 8) ---
    #: How much of the project's reference video to apply. A closed set, and
    #: ``"0"`` by default: a project with a reference attached plans exactly as
    #: it did before Phase 8 until the user turns the dial up.
    #:
    #: There is no field here for the reference itself. Which clip is the
    #: reference is project state, set through its own endpoint and resolved
    #: server-side, so a caller cannot point one request at another project's
    #: media and read its measurements back out of the plan.
    style_strength: StyleStrength = StyleStrength.ZERO

    # --- music (Phase 7) ---
    #: The track to lay under the edit. ``None`` is a silent or source-audio
    #: edit, which is every plan written before Phase 7.
    music: MusicRequest | None = None
    #: Let the track's detected beats decide clip length. Ignored when there is
    #: no music, no analysis, or analysis the server does not trust -- in which
    #: case the plan says so rather than pretending it applied.
    beat_sync: bool = False

    # --- editorial engine (Phase 11) ---
    #: Which genre policy to edit under. ``None`` derives one from the style,
    #: which is what every caller before Phase 11 effectively asked for. A
    #: closed set, so a caller names a policy and never supplies its numbers --
    #: the same argument as the aspect-ratio preset map.
    editorial_policy: PolicyId | None = None

    #: Which named alternative to produce. ``None`` is the plain edit. Because
    #: planning is deterministic, asking for the same variant twice produces the
    #: same plan -- which is what makes previewing variants without storing them
    #: safe, and what lets the client render one and then request it for real.
    variant: VariantId | None = None

    @field_validator("fps")
    @classmethod
    def _fps_must_be_a_preset(cls, value: int) -> int:
        """Only the offered frame rates.

        The plan validator permits 1-120, which is what the *renderer* can
        survive. This is the narrower question of what a client may ask for, and
        a closed set is the same argument as the aspect-ratio preset map: an
        arbitrary number here is a request nobody has reason to make.
        """
        if value not in FPS_PRESETS:
            raise ValueError(f"fps must be one of {', '.join(str(f) for f in FPS_PRESETS)}")
        return value


class ManualCutRequest(BaseModel):
    """One clip on a hand-cut timeline.

    Note what is absent, again: no ``order`` and no timeline position. The
    sequence of this list is the sequence of the edit, which is the only way to
    express it that cannot describe an overlap or a hole.
    """

    media_id: UUID
    source_in_ms: int = Field(ge=0)
    source_out_ms: int = Field(gt=0)
    transition_in: TransitionKind = TransitionKind.CUT
    #: How long the incoming transition runs. Zero for a cut, which is what the
    #: plan validator insists on.
    transition_ms: int = Field(default=0, ge=0, le=MAX_TRANSITION_MS)
    effects: list[EffectRequest] = Field(default_factory=list, max_length=MAX_EFFECTS_PER_SEGMENT)


class ManualPlanCreateRequest(BaseModel):
    """A timeline the user assembled, submitted for validation and storage.

    Carries the same output *intent* as an automatic plan -- a shape, a frame
    rate, a quality level -- and the same nothing-else. The editor sends which
    clips and which trims; it has no field for geometry, an encoder setting or a
    path, so a timeline cannot smuggle one through the route an automatic plan
    is protected from.
    """

    segments: list[ManualCutRequest] = Field(min_length=1, max_length=MAX_SEGMENTS)

    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE_16_9
    fps: int = Field(default=30, ge=1, le=60)
    fit: FitMode = FitMode.COVER
    audio: AudioMode = AudioMode.NONE
    quality: QualityPreset = QualityPreset.BALANCED
    source_gain: float = Field(default=1.0, ge=MIN_GAIN, le=MAX_GAIN)

    #: The bed the editor placed, if any. Same schema as the automatic route's,
    #: because a hand-placed cue gets no weaker a gate than a planned one.
    music: MusicRequest | None = None

    #: Subtitles the editor wrote. Validated here for shape and again by the
    #: plan validator for bounds, ordering and overlap against the real length.
    subtitles: SubtitleTrackRequest | None = None

    #: The automatic plan this timeline was cut from, when there was one.
    derived_from_edit_plan_id: UUID | None = None

    @field_validator("fps")
    @classmethod
    def _fps_must_be_a_preset(cls, value: int) -> int:
        if value not in FPS_PRESETS:
            raise ValueError(f"fps must be one of {', '.join(str(f) for f in FPS_PRESETS)}")
        return value


class EditPlanSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    planner: str
    planner_version: str
    segment_count: int
    total_duration_ms: int
    created_at: datetime


class EditPlanDetail(EditPlanSummary):
    """A plan with its full document and the selection that produced it."""

    plan: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)

    #: How the planner was chosen, and why. Present on every plan.
    mode: dict[str, Any] | None = None
    #: Provider, model, prompt version, latency, tokens, and the fallback reason
    #: if there was one. Present only when a model was involved. Never contains
    #: a key, a prompt or a completion.
    llm: dict[str, Any] | None = None

    #: The editorial account of the edit (Phase 11): the policy, the arc, each
    #: segment's narrative role, energy, beat relationship and the short reasons
    #: it was chosen, plus the eight quality metrics and every clip the engine
    #: declined. Present on plans the editorial engine produced; ``None`` on a
    #: hand-cut plan and on anything the Phase 4 rules engine made.
    #:
    #: A dictionary rather than a typed model, deliberately and for the same
    #: reason ``plan`` is one: it is a versioned document that outlives this
    #: build's schema, and a strict model here would make an older plan
    #: unreadable rather than merely unfamiliar.
    editorial: dict[str, Any] | None = None


class PlannerCapabilities(BaseModel):
    """What this server can do, for the UI to render honestly.

    Deliberately thin. The frontend needs to know whether to offer the AI option
    and what to label a plan with; it does not need, and never receives, the
    credential that makes AI possible.
    """

    ai_available: bool
    provider: str | None = None
    model: str | None = None
    #: True when the configured "provider" is the deterministic local stub. The
    #: UI says so plainly rather than letting a stub edit pass as AI.
    is_stub: bool = False
    error: str | None = None

    modes: list[str]
    styles: list[dict[str, Any]]
    aspect_ratios: list[dict[str, Any]]
    fps_presets: list[int]
    quality_presets: list[str]
    prompt_version: str
    max_request_chars: int

    #: The bounds every plan is validated against, declared rather than left to
    #: be rediscovered. An editor has to clamp a trim handle *while the pointer
    #: is moving*, which means it needs these numbers client-side; getting them
    #: from here is what stops a copy in the browser from silently drifting out
    #: of agreement with the validator that actually enforces them.
    segment_bounds: dict[str, int]

    #: The same declaration for audio: gain range, fade ceiling, cue length.
    #: A volume slider and two fade handles need clamping for exactly the same
    #: reason a trim handle does.
    audio_bounds: dict[str, float]
    #: What beat detection reports and when the planner will act on it, so the
    #: UI can explain a grid it has decided not to use rather than showing a
    #: toggle that silently does nothing.
    beat_sync: dict[str, Any]

    # ---------------------------------------------------------- Phase 9
    #: The closed transition vocabulary and its timing bounds. Declared for the
    #: same reason the segment bounds are: the editor has to grey out a
    #: transition the clips are too short for while the pointer is moving.
    transitions: list[dict[str, Any]]
    #: Every effect kind with its range and neutral value. The editor draws a
    #: slider per entry; a kind this server does not have is a slider that does
    #: not appear, rather than one that produces a 422.
    effects: list[EffectBoundsResponse]
    #: What each subtitle preset looks like, so the panel can preview it
    #: honestly instead of guessing at the font it will be rendered in.
    subtitle_styles: list[SubtitlePresetResponse]
    subtitle_positions: list[str]
    subtitle_bounds: dict[str, int]

    # ---------------------------------------------------------- Phase 10
    #: The closed operation vocabulary a co-edit may use. Declared for the same
    #: reason every other bound is: so the editor offers exactly what the server
    #: accepts, and a kind removed here stops being offered rather than becoming
    #: a 422.
    operations: list[dict[str, Any]] = Field(default_factory=list)

    # ---------------------------------------------------------- Phase 11
    #: The genre policies this build ships, with their arcs and pacing. Every
    #: entry is generated from ``domain.story``'s own table, so a policy added
    #: there appears in the UI without an edit here and one removed there stops
    #: being offered.
    editorial_policies: list[dict[str, Any]] = Field(default_factory=list)
    #: The named alternatives the variants endpoint can produce.
    variants: list[dict[str, Any]] = Field(default_factory=list)
    #: The narrative roles, in narrative order, so the editor can draw the arc
    #: without hard-coding the sequence.
    story_roles: list[str] = Field(default_factory=list)
    #: The pacing curve shapes, with their control points, so the UI can draw
    #: the curve the server will actually use rather than an illustration of it.
    pacing_shapes: list[dict[str, Any]] = Field(default_factory=list)
    #: The closed event vocabulary and the closed reason vocabulary, with their
    #: versions. The editor renders these as localised text, so it needs to know
    #: which tokens exist; a token it has no translation for is shown raw rather
    #: than hidden.
    editorial_events: list[str] = Field(default_factory=list)
    editorial_reasons: list[str] = Field(default_factory=list)
    editorial_metrics: list[dict[str, str]] = Field(default_factory=list)
    editorial_version: str = ""
    coedit_prompt_version: str = ""


class EditPlanListResponse(BaseModel):
    items: list[EditPlanSummary]
    total: int


class RenderCreateRequest(BaseModel):
    edit_plan_id: UUID


class RenderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    edit_plan_id: UUID
    job_id: UUID | None
    status: str
    bytes_size: int | None = None
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    spec: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime

    #: Presigned and short-lived; present only once the render is ready.
    playback_url: str | None = None
    playback_expires_in_s: int | None = None


class RenderListResponse(BaseModel):
    items: list[RenderResponse]
    total: int


# ------------------------------------------------------------- reference (P8)
class ReferenceRequest(BaseModel):
    """Nominate a clip as the project's style reference.

    One field, and it is an id of media in this project. Nothing about the
    reference is described by the client: what it measures as is read from the
    analysers, not asserted by the caller.
    """

    media_id: UUID


class MeasurementResponse(BaseModel):
    """A measured value and how far it should be trusted."""

    value: float
    confidence: float


class ReferenceProfileResponse(BaseModel):
    """What the reference measured as.

    Every field is nullable and every null means "not measured" -- not "zero",
    not "average". A client showing this must distinguish the two, which is why
    the confidence travels beside the value rather than being folded into it.
    """

    media_id: UUID
    version: str
    duration_ms: int
    #: Mean confidence over the features that were actually measured.
    confidence: float
    #: Whether this profile is allowed to influence an edit at all.
    usable: bool

    pacing: str | None = None
    scene_count: int | None = None
    shot_ms: MeasurementResponse | None = None
    shot_ms_p25: int | None = None
    shot_ms_p75: int | None = None
    cut_rate: MeasurementResponse | None = None

    luminance: MeasurementResponse | None = None
    contrast: MeasurementResponse | None = None
    saturation: MeasurementResponse | None = None
    motion: MeasurementResponse | None = None

    bpm: float | None = None
    beat_confidence: float | None = None
    beat_sync: MeasurementResponse | None = None
    #: True when the reference's own cuts land on its own beats often enough to
    #: be worth offering. Advisory: the server never switches beat sync on.
    suggests_beat_sync: bool = False


class ReferenceResponse(BaseModel):
    """The project's current reference, and its profile if it has been analysed."""

    media_id: UUID | None = None
    #: What the analysers still owe this reference, so a client can say "analyse
    #: it" rather than showing an empty profile with no explanation.
    pending_analyzers: list[str] = Field(default_factory=list)
    profile: ReferenceProfileResponse | None = None


# ------------------------------------------------------- Phase 9 primitives
class EffectRequest(BaseModel):
    """One effect a client asks for.

    A kind and a number, and the number's meaning comes from the kind. There is
    no parameter object and no filter field: the bounds are checked server-side
    against ``EFFECT_BOUNDS``, so a value outside them is a 422 rather than
    something the renderer has to survive.
    """

    kind: EffectKind
    amount: float
    #: Offsets within the clip, for the colour effects that support a window.
    #: Zoom and speed must span the whole clip and the plan validator says so.
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _within_bounds(self) -> EffectRequest:
        low, high, _ = EFFECT_BOUNDS[self.kind]
        if not low <= self.amount <= high:
            raise ValueError(f"{self.kind.value} must be between {low} and {high}")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be after start_ms")
        return self

    def to_domain(self) -> Effect:
        return Effect(
            kind=self.kind,
            amount=self.amount,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
        )


class SubtitleCueRequest(BaseModel):
    """One line of text, for one interval of the output.

    ``text`` is the only free string the API accepts anywhere, and it is capped,
    stripped of the characters that carry meaning to the document format, and
    stored as display content. It never becomes part of a filter expression --
    see ``infra.ffmpeg.subtitles`` for why that decision is structural rather
    than a matter of escaping carefully.
    """

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=MAX_CUE_CHARS)

    @field_validator("text")
    @classmethod
    def _plain_display_text(cls, value: str) -> str:
        cleaned = clean_text(value)
        if not cleaned:
            raise ValueError("subtitle text must contain displayable characters")
        return cleaned

    def to_domain(self) -> SubtitleCue:
        return SubtitleCue(start_ms=self.start_ms, end_ms=self.end_ms, text=self.text)


class SubtitleTrackRequest(BaseModel):
    """Every cue, and the one preset they share.

    ``style`` and ``position`` are ids. A client cannot send a font, a size, a
    colour or a coordinate -- the preset table on the server decides all of
    them, which is what keeps a font *path* out of the render.
    """

    cues: list[SubtitleCueRequest] = Field(default_factory=list, max_length=MAX_CUES)
    style: SubtitleStyle = SubtitleStyle.CLEAN
    position: SubtitlePosition = SubtitlePosition.BOTTOM

    def to_domain(self) -> SubtitleTrack:
        return SubtitleTrack(
            cues=tuple(cue.to_domain() for cue in self.cues),
            style=self.style,
            position=self.position,
        )


class SubtitlePresetResponse(BaseModel):
    """What a preset looks like, so the editor can preview it honestly.

    A description of a decision the server has already made, not a set of
    parameters the client may send back. There is no route that accepts this
    shape: a client names a preset by id and the table above decides the rest.
    """

    id: str
    label: str
    font: str
    size: int
    bold: bool


class SubtitleSuggestRequest(BaseModel):
    """Ask a model to write cues for a stored plan.

    The plan is named in the path, so the timing the model is shown is the
    timing that will be rendered. The only free text is the user's own
    description, capped at the same length the planner accepts.
    """

    request_text: str | None = Field(default=None, max_length=MAX_REQUEST_CHARS)
    style: SubtitleStyle | None = None
    position: SubtitlePosition | None = None


class SubtitleSuggestResponse(BaseModel):
    """Cues, or the reason there are none.

    ``ok`` is false and ``subtitles`` is null whenever the model could not
    produce usable cues -- there is no third state and nothing is invented to
    fill the gap. ``failure`` names which of the known failures happened so the
    panel can say "the provider is not configured" rather than "something went
    wrong".
    """

    ok: bool
    subtitles: SubtitleTrackRequest | None = None
    failure: str | None = None
    detail: str = ""
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    latency_ms: float = 0.0


# ------------------------------------------------------- co-editor (Phase 10)
class CoEditRequest(BaseModel):
    """Ask for a change to the current edit, in words or as operations.

    Two shapes, one route. ``request_text`` is the user's sentence, which the
    server resolves with its rules or hands to a model. ``operations`` is the
    same list coming back from a preview the user approved -- re-parsed and
    re-validated from scratch, because it arrived over HTTP like anything else.

    Note what a client cannot send: a plan, a timeline, a segment's media id, a
    width, an encoder setting or a path. A change request names what to change
    about the edit the server already holds.
    """

    request_text: str | None = Field(default=None, max_length=MAX_REQUEST_CHARS)
    #: Operations from a preview, when the user is confirming one. Bounded here
    #: and validated against the closed vocabulary by the domain parser.
    operations: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_OPERATIONS)
    #: The version this change was previewed against. The server refuses to
    #: apply it if the head has moved, because the clip the user meant may no
    #: longer be the clip that index names.
    base_version_id: UUID | None = None

    @model_validator(mode="after")
    def _something_to_do(self) -> CoEditRequest:
        if not self.request_text and not self.operations:
            raise ValueError("send request_text, operations, or both")
        return self


class DiffEntryResponse(BaseModel):
    """One line of the before/after the panel draws."""

    field: str
    label: str
    before: str
    after: str
    clip: int | None = None


class PlanDiffResponse(BaseModel):
    entries: list[DiffEntryResponse] = Field(default_factory=list)
    #: What the operations said they would do, in their own words.
    applied: list[str] = Field(default_factory=list)
    #: Adjustments the server made on its own to keep the plan renderable.
    #: Surfaced rather than silent.
    adjustments: list[str] = Field(default_factory=list)


class CoEditPreviewResponse(BaseModel):
    """What a change would do. Nothing has been written when this is returned.

    ``ok: false`` carries a named ``failure`` and the violations behind it --
    there is no partial state in which some of the change happened.
    """

    ok: bool
    base_version_id: UUID | None = None
    base_version: int | None = None
    operations: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    diff: PlanDiffResponse = Field(default_factory=PlanDiffResponse)
    failure: str | None = None
    detail: str = ""
    #: rules | llm | client -- which resolver read the request.
    source: str = "rules"
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    violations: list[dict[str, Any]] = Field(default_factory=list)
    #: False when the rules alone resolved it, so the editor may apply it
    #: without a confirmation step.
    needs_confirmation: bool = True


class EditVersionResponse(BaseModel):
    """One step in the history.

    No request text, only a digest: the same trade every other record of a user
    request in this API makes.
    """

    id: UUID
    version: int
    parent_id: UUID | None = None
    edit_plan_id: UUID
    origin: str
    is_current: bool
    applied: list[str] = Field(default_factory=list)
    operation_count: int = 0
    source: str = ""
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    request_digest: str | None = None
    total_duration_ms: int = 0
    segment_count: int = 0
    created_at: str
    render_id: UUID | None = None
    render_status: str | None = None


class EditVersionListResponse(BaseModel):
    items: list[EditVersionResponse]
    total: int
    current_version_id: UUID | None = None
    #: Whether the buttons should be enabled, decided by the server that owns
    #: the history rather than guessed at from the list.
    can_undo: bool = False
    can_redo: bool = False


class CoEditApplyResponse(BaseModel):
    """A committed change: the new version, its plan, and what it did."""

    version: EditVersionResponse
    plan: EditPlanDetail
    diff: PlanDiffResponse
    rationale: str = ""
    source: str = "rules"
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0


class EffectBoundsResponse(BaseModel):
    kind: str
    minimum: float
    maximum: float
    neutral: float
    #: False for the colour effects, which may be applied to part of a clip.
    whole_segment_only: bool


# ------------------------------------------------------- editorial (Phase 11)
class EditorialVariantResponse(BaseModel):
    """One editorial plan, previewed and not stored.

    A preview rather than a plan row, and that is the whole design of the
    variants endpoint: three alternatives would otherwise mean three plan rows
    and three versions per click, two of which the user never looks at again.
    Because the engine is deterministic, asking for the chosen variant through
    the ordinary plan route reproduces exactly what was previewed.
    """

    variant: str | None = None
    label: str
    description: str
    policy: str
    clip_count: int
    total_duration_ms: int
    #: The full editorial document: segments with roles, reasons and energies,
    #: the pacing plan, the metrics, and every clip the engine declined.
    editorial: dict[str, Any] = Field(default_factory=dict)


class EditorialVariantsResponse(BaseModel):
    items: list[EditorialVariantResponse] = Field(default_factory=list)


class EditorialVariantsRequest(PlanCreateRequest):
    """The same preferences as a plan request, previewed across variants.

    Subclassed rather than duplicated so the two routes cannot drift: a field
    added to planning is a field the preview honours, and a preview that
    silently ignored the target duration would be previewing a different edit
    from the one the user would get.

    ``variant`` is inherited and ignored -- the endpoint produces the base edit
    and every variant, which is what "show me the alternatives" means.
    """
