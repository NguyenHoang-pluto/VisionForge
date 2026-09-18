"""Timeline and RenderSpec: the two compilation stages between plan and FFmpeg.

    EditPlan  ->  Timeline  ->  RenderSpec  ->  argv[]
    what to      where each     what the       the actual
    include      clip sits      encoder does   command

Splitting these apart is not ceremony. Each answers a different question and
fails in a different way:

- a **Timeline** places clips on a track. Its errors are overlaps and gaps, and
  it can be checked without knowing what a codec is;
- a **RenderSpec** describes an encode. Its errors are unsupported pixel formats
  and impossible bitrates, and it can be checked without knowing what a cut is.

The Timeline is FFmpeg-free by construction: no codec, no filter, no path. That
is what will let a future manual editor mutate a timeline directly without ever
touching rendering concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visionforge.domain.editplan import (
    AspectRatio,
    AudioMode,
    EditPlan,
    Encoder,
    FitMode,
    MusicCue,
    QualityPreset,
    TransitionKind,
)
from visionforge.domain.effects import Effect, speed_of
from visionforge.domain.effects import output_duration_ms as effect_output_ms
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.subtitles import SubtitleTrack


class TrackKind(StrEnum):
    VIDEO = "video"
    #: The clips' own audio, following the video cuts exactly.
    AUDIO = "audio"
    #: An independent music bed. A separate kind rather than a second AUDIO
    #: track because the two obey different rules: source audio is cut with the
    #: picture and has no position of its own, while music has a start, a
    #: length and an envelope that are unrelated to where the cuts fall.
    MUSIC = "music"


@dataclass(frozen=True, slots=True)
class TimelineClip:
    """One clip placed on a track.

    Carries both source and timeline coordinates. The distinction matters: the
    source range says which frames to read, the timeline range says when they
    play, and conflating them is how editors end up with drift.
    """

    media_id: MediaId
    index: int
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    #: How this clip enters, and for how long (Phase 9). ``CUT`` with zero is
    #: every clip written before Phase 9.
    transition_in: TransitionKind = TransitionKind.CUT
    transition_ms: int = 0
    effects: tuple[Effect, ...] = ()

    @property
    def source_duration_ms(self) -> int:
        """The trim: how much of the source file this clip reads."""
        return self.source_out_ms - self.source_in_ms

    @property
    def duration_ms(self) -> int:
        """How long this clip plays, after any speed change.

        Not the trim. A clip at half speed reads two seconds of source and plays
        for four, and every position on the timeline is computed from the second
        number.
        """
        return effect_output_ms(self.source_duration_ms, self.effects)

    @property
    def speed(self) -> float:
        return speed_of(self.effects)

    @property
    def overlap_ms(self) -> int:
        """How much of this clip plays over the one before it."""
        return self.transition_ms if self.transition_in.consumes_time else 0

    @property
    def timeline_end_ms(self) -> int:
        return self.timeline_start_ms + self.duration_ms


@dataclass(frozen=True, slots=True)
class MusicPlacement:
    """The music cue, laid out in timeline coordinates.

    The same shape as ``TimelineClip`` -- source range plus timeline position --
    with the envelope that only audio has. Kept as its own type rather than
    reusing ``TimelineClip`` with optional fields, because a gain of ``None`` on
    a video clip is a field that exists to be ignored, and those accumulate.

    Still FFmpeg-free: a gain is a number, a fade is a duration. Nothing here
    knows what ``afade`` is called.
    """

    media_id: MediaId
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    gain: float
    fade_in_ms: int
    fade_out_ms: int

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms

    @property
    def timeline_end_ms(self) -> int:
        return self.timeline_start_ms + self.duration_ms

    def clipped_to(self, timeline_ms: int) -> MusicPlacement:
        """The cue as it will actually sound, given a timeline of this length.

        Music that outlasts the picture is normal and is accepted by the
        validator; this is where it stops. Trimming in the timeline rather than
        in the filter graph means the stored timeline shows what was heard, and
        the compiler is handed a cue that already fits.

        The fade-out is pulled back with the end, so a bed cut short still fades
        rather than stopping abruptly -- and is shortened if the truncation
        leaves no room for it.
        """
        if timeline_ms <= 0 or self.timeline_end_ms <= timeline_ms:
            return self

        audible = max(0, timeline_ms - self.timeline_start_ms)
        source_out = self.source_in_ms + audible
        fade_out = min(self.fade_out_ms, max(0, audible - self.fade_in_ms))
        return MusicPlacement(
            media_id=self.media_id,
            source_in_ms=self.source_in_ms,
            source_out_ms=source_out,
            timeline_start_ms=self.timeline_start_ms,
            gain=self.gain,
            fade_in_ms=min(self.fade_in_ms, audible),
            fade_out_ms=fade_out,
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "media_id": str(self.media_id),
            "source_in_ms": self.source_in_ms,
            "source_out_ms": self.source_out_ms,
            "timeline_start_ms": self.timeline_start_ms,
            "timeline_end_ms": self.timeline_end_ms,
            "duration_ms": self.duration_ms,
            "gain": round(self.gain, 4),
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
        }


@dataclass(frozen=True, slots=True)
class Track:
    kind: TrackKind
    clips: tuple[TimelineClip, ...]
    #: Present only on a ``MUSIC`` track. A track carries either clips or a
    #: cue, never both, which is what keeps "which track is this" answerable
    #: from the kind alone.
    music: MusicPlacement | None = None

    @property
    def duration_ms(self) -> int:
        if self.music is not None:
            return self.music.timeline_end_ms
        return max((clip.timeline_end_ms for clip in self.clips), default=0)


@dataclass(frozen=True, slots=True)
class Timeline:
    """A compiled, FFmpeg-agnostic edit.

    Contains no codec, no filter string and no filesystem path -- only what
    plays, from where, and when.
    """

    project_id: ProjectId
    tracks: tuple[Track, ...]
    width: int
    height: int
    fps: int
    aspect_ratio: AspectRatio
    fit: FitMode
    audio: AudioMode
    #: How good the output should be. Still not an encoder setting -- the level
    #: is an editorial intention, and only ``build_render_spec`` knows what it
    #: costs in CRF and preset. The timeline stays FFmpeg-free.
    quality: QualityPreset = QualityPreset.BALANCED
    encoder: Encoder = Encoder.CPU
    #: Gain on the clips' own audio. Still not an encoder setting: a number the
    #: compiler turns into a filter, the way ``quality`` becomes a CRF.
    source_gain: float = 1.0
    #: Subtitles, carried through unchanged from the plan (Phase 9). The
    #: timeline does not lay them out: cues are already in output coordinates,
    #: which is exactly why they are stored that way.
    subtitles: SubtitleTrack | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def video_track(self) -> Track:
        for track in self.tracks:
            if track.kind is TrackKind.VIDEO:
                return track
        raise ValueError("timeline has no video track")

    @property
    def music_track(self) -> Track | None:
        for track in self.tracks:
            if track.kind is TrackKind.MUSIC:
                return track
        return None

    @property
    def music(self) -> MusicPlacement | None:
        track = self.music_track
        return track.music if track else None

    @property
    def duration_ms(self) -> int:
        """How long the output is.

        The **video** track decides, not the longest track. A music bed that
        outlasts the picture does not extend the render -- there would be
        nothing to show during it -- and ``MusicPlacement.clipped_to`` has
        already cut the cue to this length by the time a timeline exists.
        """
        return self.video_track.duration_ms

    def as_payload(self) -> dict[str, Any]:
        return {
            "project_id": str(self.project_id),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "aspect_ratio": self.aspect_ratio.value,
            "fit": self.fit.value,
            "audio": self.audio.value,
            "duration_ms": self.duration_ms,
            "source_gain": round(self.source_gain, 4),
            "tracks": [
                {
                    "kind": track.kind.value,
                    "clips": [
                        {
                            "media_id": str(clip.media_id),
                            "index": clip.index,
                            "source_in_ms": clip.source_in_ms,
                            "source_out_ms": clip.source_out_ms,
                            "timeline_start_ms": clip.timeline_start_ms,
                            "timeline_end_ms": clip.timeline_end_ms,
                            "duration_ms": clip.duration_ms,
                        }
                        for clip in track.clips
                    ],
                    "music": track.music.as_payload() if track.music else None,
                }
                for track in self.tracks
            ],
            "metadata": self.metadata,
        }


def compile_timeline(plan: EditPlan) -> Timeline:
    """Lay a validated plan out on a timeline.

    Phase 4 was butt-joined and said, in this docstring, that a dissolve would
    change exactly this function -- clips overlapping by the transition duration
    with the running offset shrinking accordingly. This is that change, plus one
    the same paragraph did not anticipate: a speed effect means a clip occupies a
    different length than it reads, so the offset advances by the *output*
    duration and not by the trim.

    The two adjustments compose in one place so that the plan's
    ``total_duration_ms`` and the last clip's ``timeline_end_ms`` cannot
    disagree -- a property the tests assert directly, because an editor that
    reports one length and renders another is worse than one that refuses.

    The plan must already have passed ``assert_valid``; this compiler assumes
    well-formed input and does not re-validate.
    """
    clips: list[TimelineClip] = []
    offset = 0
    for index, segment in enumerate(plan.ordered_segments):
        # A crossfade starts this clip *before* the previous one has finished,
        # by exactly the overlap. The first clip has nothing to overlap, whatever
        # its transition claims -- validation rejects that, and this is total
        # anyway rather than trusting it to have run.
        overlap = segment.overlap_ms if index > 0 else 0
        start = max(0, offset - overlap)
        clips.append(
            TimelineClip(
                media_id=segment.media_id,
                index=index,
                source_in_ms=segment.source_in_ms,
                source_out_ms=segment.source_out_ms,
                timeline_start_ms=start,
                transition_in=segment.transition_in,
                transition_ms=segment.transition_ms,
                effects=segment.effects,
            )
        )
        offset = start + segment.output_duration_ms

    tracks = [Track(kind=TrackKind.VIDEO, clips=tuple(clips))]
    if plan.output.audio is AudioMode.SOURCE:
        # Audio follows the video cuts exactly in Phase 4: the same ranges, the
        # same offsets. A separate track rather than an implicit property of the
        # video clips, so that independent audio editing is an extension later.
        tracks.append(Track(kind=TrackKind.AUDIO, clips=tuple(clips)))

    # The extension that "later" turned into. Music is laid out against the
    # finished video length rather than against the cuts, because it has a
    # position of its own -- and is cut to that length here, so the timeline
    # records what will be heard rather than what was asked for.
    if plan.music is not None:
        tracks.append(
            Track(
                kind=TrackKind.MUSIC,
                clips=(),
                music=_place_music(plan.music, timeline_ms=offset),
            )
        )

    return Timeline(
        project_id=plan.project_id,
        tracks=tuple(tracks),
        width=plan.output.width,
        height=plan.output.height,
        fps=plan.output.fps,
        aspect_ratio=plan.output.aspect_ratio,
        fit=plan.output.fit,
        audio=plan.output.audio,
        quality=plan.output.quality,
        encoder=plan.output.encoder,
        source_gain=plan.output.source_gain,
        subtitles=plan.subtitles,
        metadata={
            "planner": plan.planner,
            "planner_version": plan.planner_version,
            "segment_count": len(clips),
            "has_music": plan.music is not None,
            "has_subtitles": plan.subtitles is not None,
            "transitions": sorted(
                {s.transition_in.value for s in plan.ordered_segments if s.transition_ms}
            ),
        },
    )


def _place_music(cue: MusicCue, *, timeline_ms: int) -> MusicPlacement:
    return MusicPlacement(
        media_id=cue.media_id,
        source_in_ms=cue.source_in_ms,
        source_out_ms=cue.source_out_ms,
        timeline_start_ms=cue.timeline_start_ms,
        gain=cue.gain,
        fade_in_ms=cue.fade_in_ms,
        fade_out_ms=cue.fade_out_ms,
    ).clipped_to(timeline_ms)


# --------------------------------------------------------------- render spec
#: The largest side NVENC's H.264 encoder accepts.
NVENC_H264_MAX = 4096


class VideoCodec(StrEnum):
    H264 = "h264"
    #: Used by the GPU encoder above 4096 a side, which NVENC's H.264 refuses.
    HEVC = "hevc"


class AudioCodec(StrEnum):
    AAC = "aac"


class Container(StrEnum):
    MP4 = "mp4"


@dataclass(frozen=True, slots=True)
class RenderInput:
    """One decoded source, identified by id and resolved to a local path later.

    ``local_path`` is filled by the worker from the storage key it derived
    itself. It is never supplied by a client, and never travels in a plan.
    """

    media_id: MediaId
    index: int
    local_path: str
    #: For a still (Phase 12), how long the looped picture must run. ``None``
    #: for a video or audio file, which bring their own timeline.
    still_ms: int | None = None

    @property
    def is_still(self) -> bool:
        return self.still_ms is not None


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """One trimmed piece of one input, in the order it will be joined."""

    input_index: int
    source_in_ms: int
    source_out_ms: int
    #: How this segment enters, and for how long. ``CUT`` with zero is every
    #: segment written before Phase 9, and the compiler keeps its old code path
    #: for a spec in which every segment says that.
    transition_in: TransitionKind = TransitionKind.CUT
    transition_ms: int = 0
    effects: tuple[Effect, ...] = ()

    @property
    def source_duration_ms(self) -> int:
        """The trim, in source time. What ``atrim``/``trim`` are given."""
        return self.source_out_ms - self.source_in_ms

    @property
    def duration_ms(self) -> int:
        """How long this segment plays, after any speed change."""
        return effect_output_ms(self.source_duration_ms, self.effects)

    @property
    def speed(self) -> float:
        return speed_of(self.effects)

    @property
    def overlap_ms(self) -> int:
        return self.transition_ms if self.transition_in.consumes_time else 0


@dataclass(frozen=True, slots=True)
class RenderMusic:
    """The music bed, resolved to an input index and a local file.

    ``input_index`` points into ``RenderSpec.inputs``, so the music is an
    ordinary FFmpeg input like any clip -- it just happens to be the only one
    whose audio is used without its video. ``local_path`` is filled by the
    worker from a storage key it derived itself, exactly as for a video source.
    """

    input_index: int
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    gain: float
    fade_in_ms: int
    fade_out_ms: int

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Everything the encoder needs, and nothing it does not.

    **No shell strings live here.** The fields are typed values; turning them
    into a filter graph and an argv array is the compiler's job, and it is the
    only code that knows FFmpeg's syntax.
    """

    inputs: tuple[RenderInput, ...]
    segments: tuple[RenderSegment, ...]
    width: int
    height: int
    fps: int
    fit: FitMode
    output_path: str
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    container: Container = Container.MP4
    #: Constant Rate Factor. 23 is FFmpeg's default and visually transparent
    #: enough for a preview render; lower would cost encode time this machine
    #: does not have spare.
    crf: int = 23
    preset: str = "veryfast"
    pixel_format: str = "yuv420p"
    audio_bitrate_kbps: int = 128
    #: Which hardware encodes. ``crf`` and ``preset`` are that encoder's own.
    encoder: Encoder = Encoder.CPU
    #: Resampling algorithm for every scale, or ``None`` for FFmpeg's default.
    scale_flags: str | None = None
    #: Scale a zoomed or panned segment *larger* than the output before
    #: cropping, so the crop is taken from real pixels rather than upscaled
    #: back into the frame. Off for the fast levels, whose output is unchanged.
    supersample: bool = False
    #: Whether the clips' own audio is concatenated into the output.
    include_audio: bool = False
    #: Gain applied to that source audio. Ignored when it is not included.
    source_gain: float = 1.0
    #: The music bed, if there is one. Independent of ``include_audio``: music
    #: alone, source alone, both, or neither are all valid outputs.
    music: RenderMusic | None = None
    #: Subtitles, if any (Phase 9). The compiler turns these into an ASS
    #: document in the render's own scratch directory; the spec carries no path
    #: because the path does not exist until the compiler makes one.
    subtitles: SubtitleTrack | None = None

    @property
    def duration_ms(self) -> int:
        """How long the encode will run.

        A sum minus the overlaps. Phase 4 could sum the segments because every
        join was a cut; a crossfade makes the output shorter than its parts, and
        a spec that still reported the sum would make the progress bar, the
        music trim and the stored duration all wrong in the same direction.
        """
        if not self.segments:
            return 0
        total = sum(segment.duration_ms for segment in self.segments)
        return total - sum(segment.overlap_ms for segment in self.segments[1:])

    @property
    def has_transitions(self) -> bool:
        """Whether any join is something other than a hard cut.

        The compiler branches on this: a spec of pure cuts takes the Phase 4
        concat path unchanged, so eight phases of existing plans render to the
        same bytes they always did.
        """
        return any(segment.transition_in is not TransitionKind.CUT for segment in self.segments)

    @property
    def has_effects(self) -> bool:
        return any(segment.effects for segment in self.segments)

    @property
    def has_audio_output(self) -> bool:
        """Whether the encode produces an audio stream at all."""
        return self.include_audio or self.music is not None

    def as_payload(self) -> dict[str, Any]:
        """Operational summary. Local paths are omitted deliberately.

        The spec is stored on the render row and returned by the API; a
        filesystem path is worker-local, meaningless to a client, and not
        something to hand out.
        """
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fit": self.fit.value,
            "video_codec": self.video_codec.value,
            "audio_codec": self.audio_codec.value if self.has_audio_output else None,
            "container": self.container.value,
            "crf": self.crf,
            "preset": self.preset,
            "pixel_format": self.pixel_format,
            "input_count": len(self.inputs),
            "segment_count": len(self.segments),
            "duration_ms": self.duration_ms,
            "source_audio": self.include_audio,
            "source_gain": round(self.source_gain, 4) if self.include_audio else None,
            # What was mixed, not where it came from: a storage key is
            # worker-local and is not something to hand out on a render row.
            "music": (
                {
                    "duration_ms": self.music.duration_ms,
                    "timeline_start_ms": self.music.timeline_start_ms,
                    "gain": round(self.music.gain, 4),
                    "fade_in_ms": self.music.fade_in_ms,
                    "fade_out_ms": self.music.fade_out_ms,
                }
                if self.music
                else None
            ),
        }


def build_render_spec(
    timeline: Timeline,
    *,
    local_paths: dict[MediaId, str],
    output_path: str,
    still_ids: frozenset[MediaId] = frozenset(),
) -> RenderSpec:
    """Turn a timeline into an encoder-ready spec.

    ``local_paths`` is supplied by the worker, which resolved each media id to a
    storage key and downloaded it. Nothing here accepts a path from anywhere
    else, which is what keeps arbitrary filesystem access out of the render path.

    ``still_ids`` names the media that are photos (Phase 12). The worker knows
    that from the media rows; a plan does not carry it. A still is decoded as a
    looped picture, and has no audio stream to take sound from.

    Every segment is an input of its own, a clip used twice included; see the
    comment below for why that is memory rather than waste.
    """
    clips = timeline.video_track.clips
    music = timeline.music
    needed = [clip.media_id for clip in clips]
    if music is not None:
        needed.append(music.media_id)
    missing = [media_id for media_id in needed if media_id not in local_paths]
    if missing:
        raise ValueError(f"no local path resolved for media: {missing}")

    # One input per segment, even when a clip is used twice. Sharing one decoded
    # stream between two segments makes FFmpeg queue every frame the later
    # segment needs, already scaled to the output, until the edit reaches it --
    # at 4K that is over a gigabyte for three seconds of one reused photo, and
    # templates reuse clips by design. Opening the file again costs a second
    # decode; queueing it costs the machine's memory.
    inputs: list[RenderInput] = [
        RenderInput(
            media_id=clip.media_id,
            index=position,
            local_path=local_paths[clip.media_id],
            # Long enough for this hold. A hold starts at zero, so that is
            # its out point.
            still_ms=clip.source_out_ms if clip.media_id in still_ids else None,
        )
        for position, clip in enumerate(clips)
    ]

    segments = tuple(
        RenderSegment(
            input_index=position,
            source_in_ms=clip.source_in_ms,
            source_out_ms=clip.source_out_ms,
            transition_in=clip.transition_in,
            transition_ms=clip.transition_ms,
            effects=clip.effects,
        )
        for position, clip in enumerate(clips)
    )

    # The music file is an input of its own, after every clip.
    render_music: RenderMusic | None = None
    if music is not None:
        inputs.append(
            RenderInput(
                media_id=music.media_id,
                index=len(inputs),
                local_path=local_paths[music.media_id],
            )
        )
        render_music = RenderMusic(
            input_index=len(inputs) - 1,
            source_in_ms=music.source_in_ms,
            source_out_ms=music.source_out_ms,
            timeline_start_ms=music.timeline_start_ms,
            gain=music.gain,
            fade_in_ms=music.fade_in_ms,
            fade_out_ms=music.fade_out_ms,
        )

    # The only place a quality *level* becomes encoder *settings*. Imported
    # here rather than at module scope because ``style`` imports ``editplan``,
    # and a top-level import would make the domain's dependency graph circular.
    from visionforge.domain.style import (
        AUDIO_KBPS,
        GPU_QUALITY_SETTINGS,
        QUALITY_SETTINGS,
        SCALE_FLAGS,
    )

    gpu = timeline.encoder is Encoder.GPU
    crf, preset = (GPU_QUALITY_SETTINGS if gpu else QUALITY_SETTINGS)[timeline.quality]
    # NVENC's H.264 stops at 4096 a side; its HEVC goes to 8192.
    codec = (
        VideoCodec.HEVC
        if gpu and max(timeline.width, timeline.height) > NVENC_H264_MAX
        else VideoCodec.H264
    )
    scale_flags = SCALE_FLAGS[timeline.quality]

    return RenderSpec(
        inputs=tuple(inputs),
        segments=segments,
        width=timeline.width,
        height=timeline.height,
        fps=timeline.fps,
        fit=timeline.fit,
        output_path=output_path,
        crf=crf,
        preset=preset,
        encoder=timeline.encoder,
        video_codec=codec,
        audio_bitrate_kbps=AUDIO_KBPS[timeline.quality],
        scale_flags=scale_flags,
        supersample=scale_flags is not None,
        include_audio=timeline.audio is AudioMode.SOURCE,
        source_gain=timeline.source_gain,
        music=render_music,
        # Carried through from the timeline, which carried it from the plan. It
        # was missing here once, and nothing caught it until a real render
        # produced a file whose frames were identical with and without the
        # track -- the compiler maps [vout] rather than [vsub] when the spec
        # says there are no subtitles, and it was telling the truth.
        subtitles=timeline.subtitles,
    )


__all__ = [
    "AudioCodec",
    "Container",
    "MusicPlacement",
    "RenderInput",
    "RenderMusic",
    "RenderSegment",
    "RenderSpec",
    "Timeline",
    "TimelineClip",
    "Track",
    "TrackKind",
    "VideoCodec",
    "build_render_spec",
    "compile_timeline",
]
