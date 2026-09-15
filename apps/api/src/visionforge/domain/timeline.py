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
    FitMode,
    MusicCue,
    QualityPreset,
)
from visionforge.domain.ids import MediaId, ProjectId


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

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms

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
    #: Gain on the clips' own audio. Still not an encoder setting: a number the
    #: compiler turns into a filter, the way ``quality`` becomes a CRF.
    source_gain: float = 1.0
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

    Butt-joined: each clip starts where the previous ended. Phase 4 has hard
    cuts only, so there is no overlap to negotiate and no gap to fill. When a
    dissolve arrives it changes exactly this function -- clips overlap by the
    transition duration and the running offset shrinks accordingly.

    The plan must already have passed ``assert_valid``; this compiler assumes
    well-formed input and does not re-validate.
    """
    clips: list[TimelineClip] = []
    offset = 0
    for index, segment in enumerate(plan.ordered_segments):
        clips.append(
            TimelineClip(
                media_id=segment.media_id,
                index=index,
                source_in_ms=segment.source_in_ms,
                source_out_ms=segment.source_out_ms,
                timeline_start_ms=offset,
            )
        )
        offset += segment.duration_ms

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
        source_gain=plan.output.source_gain,
        metadata={
            "planner": plan.planner,
            "planner_version": plan.planner_version,
            "segment_count": len(clips),
            "has_music": plan.music is not None,
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
class VideoCodec(StrEnum):
    H264 = "h264"


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


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """One trimmed piece of one input, in the order it will be concatenated."""

    input_index: int
    source_in_ms: int
    source_out_ms: int

    @property
    def duration_ms(self) -> int:
        return self.source_out_ms - self.source_in_ms


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
    #: Whether the clips' own audio is concatenated into the output.
    include_audio: bool = False
    #: Gain applied to that source audio. Ignored when it is not included.
    source_gain: float = 1.0
    #: The music bed, if there is one. Independent of ``include_audio``: music
    #: alone, source alone, both, or neither are all valid outputs.
    music: RenderMusic | None = None

    @property
    def duration_ms(self) -> int:
        return sum(segment.duration_ms for segment in self.segments)

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
) -> RenderSpec:
    """Turn a timeline into an encoder-ready spec.

    ``local_paths`` is supplied by the worker, which resolved each media id to a
    storage key and downloaded it. Nothing here accepts a path from anywhere
    else, which is what keeps arbitrary filesystem access out of the render path.

    Distinct media become distinct inputs; a clip used twice is one input with
    two segments, so the file is decoded once.
    """
    clips = timeline.video_track.clips
    music = timeline.music
    needed = [clip.media_id for clip in clips]
    if music is not None:
        needed.append(music.media_id)
    missing = [media_id for media_id in needed if media_id not in local_paths]
    if missing:
        raise ValueError(f"no local path resolved for media: {missing}")

    input_index: dict[MediaId, int] = {}
    inputs: list[RenderInput] = []
    for clip in clips:
        if clip.media_id not in input_index:
            input_index[clip.media_id] = len(inputs)
            inputs.append(
                RenderInput(
                    media_id=clip.media_id,
                    index=len(inputs),
                    local_path=local_paths[clip.media_id],
                )
            )

    segments = tuple(
        RenderSegment(
            input_index=input_index[clip.media_id],
            source_in_ms=clip.source_in_ms,
            source_out_ms=clip.source_out_ms,
        )
        for clip in clips
    )

    # The music file becomes an input like any other. If the same asset were
    # somehow also a video source it would be decoded once and used twice,
    # which is the same deduplication the clips get.
    render_music: RenderMusic | None = None
    if music is not None:
        if music.media_id not in input_index:
            input_index[music.media_id] = len(inputs)
            inputs.append(
                RenderInput(
                    media_id=music.media_id,
                    index=len(inputs),
                    local_path=local_paths[music.media_id],
                )
            )
        render_music = RenderMusic(
            input_index=input_index[music.media_id],
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
    from visionforge.domain.style import QUALITY_SETTINGS

    crf, preset = QUALITY_SETTINGS[timeline.quality]

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
        include_audio=timeline.audio is AudioMode.SOURCE,
        source_gain=timeline.source_gain,
        music=render_music,
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
