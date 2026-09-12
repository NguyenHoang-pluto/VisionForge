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

from visionforge.domain.editplan import AspectRatio, AudioMode, EditPlan, FitMode
from visionforge.domain.ids import MediaId, ProjectId


class TrackKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


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
class Track:
    kind: TrackKind
    clips: tuple[TimelineClip, ...]

    @property
    def duration_ms(self) -> int:
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def video_track(self) -> Track:
        for track in self.tracks:
            if track.kind is TrackKind.VIDEO:
                return track
        raise ValueError("timeline has no video track")

    @property
    def duration_ms(self) -> int:
        return max((track.duration_ms for track in self.tracks), default=0)

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

    return Timeline(
        project_id=plan.project_id,
        tracks=tuple(tracks),
        width=plan.output.width,
        height=plan.output.height,
        fps=plan.output.fps,
        aspect_ratio=plan.output.aspect_ratio,
        fit=plan.output.fit,
        audio=plan.output.audio,
        metadata={
            "planner": plan.planner,
            "planner_version": plan.planner_version,
            "segment_count": len(clips),
        },
    )


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
    include_audio: bool = False

    @property
    def duration_ms(self) -> int:
        return sum(segment.duration_ms for segment in self.segments)

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
            "audio_codec": self.audio_codec.value if self.include_audio else None,
            "container": self.container.value,
            "crf": self.crf,
            "preset": self.preset,
            "pixel_format": self.pixel_format,
            "input_count": len(self.inputs),
            "segment_count": len(self.segments),
            "duration_ms": self.duration_ms,
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
    missing = [clip.media_id for clip in clips if clip.media_id not in local_paths]
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

    return RenderSpec(
        inputs=tuple(inputs),
        segments=segments,
        width=timeline.width,
        height=timeline.height,
        fps=timeline.fps,
        fit=timeline.fit,
        output_path=output_path,
        include_audio=timeline.audio is AudioMode.SOURCE,
    )


__all__ = [
    "AudioCodec",
    "Container",
    "RenderInput",
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
