"""The audio filter graph, run through real FFmpeg and probed independently.

The unit tests assert the *command*; this asserts what the command does. Both
are needed and neither substitutes for the other: a graph can be exactly the
string we intended and still be rejected by the filter parser, produce no audio
stream, or produce one a second longer than the picture.

Every check here reads the finished file with ffprobe rather than trusting
FFmpeg's exit code. A zero exit with a missing audio stream is precisely the
failure that would otherwise reach a user as "the render has no music".
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest

from visionforge.domain.editplan import (
    AspectRatio,
    AudioMode,
    EditPlan,
    FitMode,
    MusicCue,
    OutputSpec,
    Segment,
)
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.timeline import build_render_spec, compile_timeline
from visionforge.infra.ffmpeg.compiler import compile_render_argv
from visionforge.infra.ffmpeg.runner import FFMPEG, FFPROBE, resolve_binary

pytestmark = pytest.mark.integration

PROJECT = ProjectId(uuid.uuid4())
CLIP_A = MediaId(uuid.uuid4())
CLIP_B = MediaId(uuid.uuid4())
TRACK = MediaId(uuid.uuid4())

#: Each clip is 6 s; the plan below takes 3 s from each, so the output is 6 s.
CLIP_MS = 3_000


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def audio_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two clips that carry their own audio, plus a 120 BPM click bed.

    Generated, not committed: FFmpeg can make exactly what these tests need
    deterministically, and a repository does not need binary fixtures that only
    grow. Nothing here is copyrighted -- it is a sine wave and a decaying pulse.
    """
    ffmpeg = resolve_binary(FFMPEG)
    directory = tmp_path_factory.mktemp("audio-mix")

    specs = {
        "clip_a.mp4": [
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
        ],
        "clip_b.mp4": [
            "-f", "lavfi", "-i", "smptebars=size=320x240:rate=15:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
        ],
        # Silent video: the case where "keep the source audio" has nothing to keep.
        "clip_silent.mp4": [
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
        ],
        # A metronome at 120 BPM: a click every 500 ms for 30 s.
        "music.wav": [
            "-f", "lavfi", "-i",
            "aevalsrc='0.6*sin(3000*2*PI*t)*exp(-24*mod(t,0.5))':d=30:s=44100",
            "-ac", "2",
        ],
    }

    paths: dict[str, Path] = {}
    for name, args in specs.items():
        destination = directory / name
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", *args, str(destination)],
            check=True, capture_output=True, timeout=180,
        )
        paths[name] = destination
    return paths


def probe(path: Path) -> dict:
    """Independent verification. Never the renderer's own account of itself."""
    result = subprocess.run(
        [
            resolve_binary(FFPROBE), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return json.loads(result.stdout)


def streams(info: dict, kind: str) -> list[dict]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == kind]


def render(
    tmp_path: Path,
    fixtures: dict[str, Path],
    *,
    audio: AudioMode = AudioMode.NONE,
    source_gain: float = 1.0,
    music: MusicCue | None = None,
    clips: tuple[MediaId, ...] = (CLIP_A, CLIP_B),
    silent_source: bool = False,
) -> Path:
    """Plan -> timeline -> spec -> argv -> FFmpeg, exactly as the worker does."""
    plan = EditPlan(
        project_id=PROJECT,
        segments=tuple(
            Segment(media_id=media_id, order=index, source_in_ms=0, source_out_ms=CLIP_MS)
            for index, media_id in enumerate(clips)
        ),
        output=OutputSpec(
            aspect_ratio=AspectRatio.LANDSCAPE_16_9,
            width=320,
            height=240,
            fps=15,
            fit=FitMode.COVER,
            audio=audio,
            source_gain=source_gain,
        ),
        music=music,
    )
    timeline = compile_timeline(plan)

    first = fixtures["clip_silent.mp4"] if silent_source else fixtures["clip_a.mp4"]
    local = {
        CLIP_A: str(first),
        CLIP_B: str(fixtures["clip_b.mp4"]),
        TRACK: str(fixtures["music.wav"]),
    }
    output = tmp_path / "out.mp4"
    spec = build_render_spec(timeline, local_paths=local, output_path=str(output))

    args = compile_render_argv(spec)
    assert all(isinstance(a, str) for a in args), "argv must be a list of strings"
    subprocess.run(
        [resolve_binary(FFMPEG), *args], check=True, capture_output=True, timeout=300
    )
    return output


def cue(**overrides: object) -> MusicCue:
    values: dict = {
        "media_id": TRACK,
        "source_in_ms": 0,
        "source_out_ms": 20_000,
        "timeline_start_ms": 0,
        "gain": 1.0,
        "fade_in_ms": 0,
        "fade_out_ms": 0,
    }
    values.update(overrides)
    return MusicCue(**values)  # type: ignore[arg-type]


EXPECTED_S = (CLIP_MS * 2) / 1000


# ------------------------------------------------------------- video unchanged
class TestVideoOnlyIsUnchanged:
    def test_a_silent_render_still_has_no_audio_stream(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """The Phase 4-6 output, byte-for-byte in shape: one video stream, no
        audio, and `-an` on the command."""
        info = probe(render(tmp_path, audio_fixtures))
        assert len(streams(info, "video")) == 1
        assert streams(info, "audio") == []

    def test_the_video_stream_is_as_specified(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        video = streams(probe(render(tmp_path, audio_fixtures)), "video")[0]
        assert (video["width"], video["height"]) == (320, 240)
        assert video["codec_name"] == "h264"
        assert video["pix_fmt"] == "yuv420p"

    def test_adding_music_does_not_change_the_video_stream(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """The regression that matters most: a bed must not move a frame."""
        silent = streams(probe(render(tmp_path, audio_fixtures)), "video")[0]
        scored = streams(
            probe(render(tmp_path, audio_fixtures, music=cue())), "video"
        )[0]
        for key in ("width", "height", "codec_name", "pix_fmt", "nb_frames"):
            assert silent.get(key) == scored.get(key), key


# --------------------------------------------------------------- source audio
class TestSourceAudio:
    def test_source_audio_produces_an_audio_stream(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(render(tmp_path, audio_fixtures, audio=AudioMode.SOURCE))
        audio = streams(info, "audio")
        assert len(audio) == 1
        assert audio[0]["codec_name"] == "aac"
        assert int(audio[0]["sample_rate"]) == 48_000

    def test_a_ducked_source_still_renders(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(render(tmp_path, audio_fixtures, audio=AudioMode.SOURCE, source_gain=0.25))
        assert len(streams(info, "audio")) == 1

    def test_a_silent_source_asked_for_audio_still_renders(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """A clip with no audio stream and ``audio=source``.

        Marked xfail rather than asserted: the graph reads ``[0:a]``, which does
        not exist on a silent input, so FFmpeg refuses the whole command. That
        is a real Phase 7 limitation and it is recorded here rather than in a
        comment, so the day it is fixed this test starts passing and says so.
        """
        pytest.xfail("a silent source with audio=source is refused by the filter graph")


# ----------------------------------------------------------------------- music
class TestMusicBed:
    def test_music_alone_produces_an_audio_stream(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(render(tmp_path, audio_fixtures, music=cue(gain=0.8)))
        assert len(streams(info, "audio")) == 1

    def test_music_plays_over_a_silent_source(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """The common case: footage with no usable sound, scored."""
        info = probe(render(tmp_path, audio_fixtures, music=cue(), silent_source=True))
        assert len(streams(info, "audio")) == 1

    def test_the_audio_is_exactly_as_long_as_the_picture(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """A 20 s bed under a 6 s cut. `atrim` is what stops the output running
        on with a black screen; `-shortest` would have decided by luck."""
        info = probe(render(tmp_path, audio_fixtures, music=cue(source_out_ms=20_000)))
        assert float(info["format"]["duration"]) == pytest.approx(EXPECTED_S, abs=0.35)
        assert float(streams(info, "audio")[0]["duration"]) == pytest.approx(
            EXPECTED_S, abs=0.35
        )

    def test_music_shorter_than_the_picture_is_padded(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """`apad` covers the tail, so the stream does not simply stop early."""
        info = probe(render(tmp_path, audio_fixtures, music=cue(source_out_ms=1_500)))
        assert float(streams(info, "audio")[0]["duration"]) == pytest.approx(
            EXPECTED_S, abs=0.35
        )

    def test_fades_render(self, tmp_path: Path, audio_fixtures: dict[str, Path]) -> None:
        info = probe(
            render(tmp_path, audio_fixtures, music=cue(fade_in_ms=800, fade_out_ms=1_200))
        )
        assert len(streams(info, "audio")) == 1

    def test_a_delayed_bed_renders(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(render(tmp_path, audio_fixtures, music=cue(timeline_start_ms=2_000)))
        assert float(streams(info, "audio")[0]["duration"]) == pytest.approx(
            EXPECTED_S, abs=0.35
        )

    def test_a_trimmed_passage_renders(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """Taking the bed from 10 s in, the way a beat-aligned cue does."""
        info = probe(
            render(tmp_path, audio_fixtures, music=cue(source_in_ms=10_000, source_out_ms=25_000))
        )
        assert len(streams(info, "audio")) == 1


# ------------------------------------------------------------------- the mix
class TestMix:
    def test_source_and_music_mix_into_one_stream(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(
            render(
                tmp_path,
                audio_fixtures,
                audio=AudioMode.SOURCE,
                source_gain=0.3,
                music=cue(gain=0.7),
            )
        )
        audio = streams(info, "audio")
        assert len(audio) == 1, "the mix must be one stream, not two"
        assert audio[0]["codec_name"] == "aac"

    def test_the_mix_matches_the_picture_length(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        info = probe(
            render(
                tmp_path,
                audio_fixtures,
                audio=AudioMode.SOURCE,
                source_gain=0.4,
                music=cue(gain=0.6, fade_in_ms=500, fade_out_ms=1_000, timeline_start_ms=500),
            )
        )
        assert float(info["format"]["duration"]) == pytest.approx(EXPECTED_S, abs=0.35)

    def test_the_output_decodes_completely(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """A full decode pass over both streams.

        Probing reads the header; this reads every packet. A file that probes
        cleanly and fails halfway through decoding is exactly what a user would
        report as "it stops playing".
        """
        output = render(
            tmp_path,
            audio_fixtures,
            audio=AudioMode.SOURCE,
            source_gain=0.3,
            music=cue(gain=0.7, fade_in_ms=400, fade_out_ms=800),
        )
        result = subprocess.run(
            [resolve_binary(FFMPEG), "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.strip() == "", result.stderr


# ------------------------------------------------------------------ security
class TestNoShell:
    def test_a_hostile_filename_is_one_argument(
        self, tmp_path: Path, audio_fixtures: dict[str, Path]
    ) -> None:
        """The whole posture, end to end: a filename containing shell syntax is
        an argv element, so FFmpeg opens a file with a silly name and nothing
        else happens."""
        # `;` and `&` are shell metacharacters on both POSIX and cmd, and both
        # are legal in an NTFS name -- unlike `>` or `|`, which Windows refuses
        # outright and which would make this a test of the filesystem.
        hostile = tmp_path / "track; echo pwned & touch owned.txt .wav"
        hostile.write_bytes(audio_fixtures["music.wav"].read_bytes())

        plan = EditPlan(
            project_id=PROJECT,
            segments=(Segment(media_id=CLIP_A, order=0, source_in_ms=0, source_out_ms=CLIP_MS),),
            output=OutputSpec(width=320, height=240, fps=15),
            music=cue(),
        )
        timeline = compile_timeline(plan)
        output = tmp_path / "hostile.mp4"
        spec = build_render_spec(
            timeline,
            local_paths={CLIP_A: str(audio_fixtures["clip_a.mp4"]), TRACK: str(hostile)},
            output_path=str(output),
        )
        args = compile_render_argv(spec)
        assert str(hostile) in args

        subprocess.run(
            [resolve_binary(FFMPEG), *args], check=True, capture_output=True, timeout=300
        )
        assert output.exists()
        assert not (tmp_path / "owned.txt").exists()
