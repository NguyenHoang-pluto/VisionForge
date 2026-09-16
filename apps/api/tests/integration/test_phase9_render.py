"""Transitions, effects and subtitles against real FFmpeg.

The unit tests prove the arithmetic. These prove the thing arithmetic cannot:
that the filter strings this compiler emits are accepted by FFmpeg, and that the
file which comes out is the length the domain said it would be.

Both halves matter. A graph that fails to parse is a loud bug; a graph that
parses and produces a video of the wrong length is a quiet one, and quiet is
what this suite is for. Every case asserts the *measured* duration against the
spec's own claim.

No database and no object storage: these are FFmpeg tests, and the fixtures are
generated with FFmpeg into a temporary directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from visionforge.domain.editplan import FitMode, TransitionKind
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId
from visionforge.domain.subtitles import (
    SubtitleCue,
    SubtitlePosition,
    SubtitleStyle,
    SubtitleTrack,
)
from visionforge.domain.timeline import RenderInput, RenderSegment, RenderSpec
from visionforge.infra.ffmpeg.compiler import compile_render_argv
from visionforge.infra.ffmpeg.subtitles import build_ass

pytestmark = pytest.mark.integration

#: How far a rendered file may differ from the duration the spec claims.
#:
#: One frame at 25 fps is 40 ms; `xfade` and the MP4 muxer each round to a frame
#: boundary, so two frames of slack is agreement rather than tolerance of a bug.
DURATION_TOLERANCE_MS = 120


def _binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:  # pragma: no cover - environment guard
        pytest.skip(f"{name} not on PATH")
    return found


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two visually distinct silent clips, and one with an audio track."""
    ffmpeg = _binary("ffmpeg")
    directory = tmp_path_factory.mktemp("phase9")

    # All three carry audio. A spec with `include_audio` against a silent
    # source is a different scenario with its own handling, and mixing the two
    # concerns in one fixture made a real timebase bug look like a missing
    # stream.
    specs = {
        "a": ("testsrc", True),
        "b": ("smptebars", True),
        "sound": ("testsrc", True),
    }
    made: dict[str, Path] = {}
    for name, (pattern, with_audio) in specs.items():
        path = directory / f"{name}.mp4"
        args = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=size=320x240:rate=25:duration=4",
        ]
        if with_audio:
            args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac"]
        else:
            args += ["-an"]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(args, check=True, capture_output=True, timeout=180)
        made[name] = path

    return made


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            _binary("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def measured_ms(path: Path) -> int:
    return int(float(probe(path)["format"]["duration"]) * 1000)


def render(spec: RenderSpec, *, ass_path: str | None = None) -> None:
    """Run the compiled command and fail with FFmpeg's own words if it errors."""
    argv = compile_render_argv(spec, ass_path=ass_path)
    result = subprocess.run([_binary("ffmpeg"), *argv], capture_output=True, timeout=900)
    if result.returncode != 0:
        raise AssertionError(
            "ffmpeg failed\n"
            + result.stderr.decode(errors="replace")[-1500:]
            + "\n\nfiltergraph:\n"
            + argv[argv.index("-filter_complex") + 1]
        )


def spec_for(
    clips: dict[str, Path],
    segments: tuple[RenderSegment, ...],
    output: Path,
    **kwargs: object,
) -> RenderSpec:
    inputs = tuple(
        RenderInput(media_id=MediaId(uuid.uuid4()), index=index, local_path=str(path))
        for index, path in enumerate([clips["a"], clips["b"], clips["sound"]])
    )
    return RenderSpec(
        inputs=inputs,
        segments=segments,
        width=320,
        height=240,
        fps=25,
        fit=FitMode.COVER,
        output_path=str(output),
        preset="ultrafast",
        crf=30,
        **kwargs,  # type: ignore[arg-type]
    )


def assert_renders_to_claimed_length(spec: RenderSpec, *, ass_path: str | None = None) -> int:
    render(spec, ass_path=ass_path)
    output = Path(spec.output_path)
    assert output.exists() and output.stat().st_size > 0

    actual = measured_ms(output)
    assert (
        abs(actual - spec.duration_ms) <= DURATION_TOLERANCE_MS
    ), f"claimed {spec.duration_ms} ms, rendered {actual} ms"
    return actual


class TestTransitions:
    def test_plain_cuts_still_render(self, clips: dict[str, Path], workdir: Path) -> None:
        """The Phase 4 path, which a spec of pure cuts must still take."""
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 2_000), RenderSegment(1, 0, 2_000)),
            workdir / "cuts.mp4",
        )
        assert not spec.has_transitions
        assert_renders_to_claimed_length(spec)

    def test_a_crossfade_shortens_the_output(self, clips: dict[str, Path], workdir: Path) -> None:
        spec = spec_for(
            clips,
            (
                RenderSegment(0, 0, 2_000),
                RenderSegment(
                    1, 0, 2_000, transition_in=TransitionKind.CROSSFADE, transition_ms=500
                ),
            ),
            workdir / "xfade.mp4",
        )
        assert spec.has_transitions
        assert spec.duration_ms == 3_500
        assert_renders_to_claimed_length(spec)

    def test_several_crossfades_chain(self, clips: dict[str, Path], workdir: Path) -> None:
        """Each `xfade` offset is measured from the accumulated output, so an
        error in the arithmetic compounds rather than cancelling."""
        spec = spec_for(
            clips,
            (
                RenderSegment(0, 0, 2_000),
                RenderSegment(
                    1, 0, 2_000, transition_in=TransitionKind.CROSSFADE, transition_ms=400
                ),
                RenderSegment(
                    0, 1_000, 3_000, transition_in=TransitionKind.CROSSFADE, transition_ms=600
                ),
            ),
            workdir / "xfade3.mp4",
        )
        assert spec.duration_ms == 6_000 - 400 - 600
        assert_renders_to_claimed_length(spec)

    def test_fades_do_not_change_the_length(self, clips: dict[str, Path], workdir: Path) -> None:
        spec = spec_for(
            clips,
            (
                RenderSegment(0, 0, 2_000, transition_in=TransitionKind.FADE_IN, transition_ms=500),
                RenderSegment(
                    1, 0, 2_000, transition_in=TransitionKind.FADE_TO_BLACK, transition_ms=500
                ),
            ),
            workdir / "fades.mp4",
        )
        assert spec.duration_ms == 4_000
        assert_renders_to_claimed_length(spec)

    def test_a_cut_and_a_crossfade_mix_in_one_edit(
        self, clips: dict[str, Path], workdir: Path
    ) -> None:
        """The pairwise chain has to handle both kinds of join in one graph."""
        spec = spec_for(
            clips,
            (
                RenderSegment(0, 0, 1_500),
                RenderSegment(1, 0, 1_500),
                RenderSegment(
                    0, 2_000, 3_500, transition_in=TransitionKind.CROSSFADE, transition_ms=400
                ),
            ),
            workdir / "mixed.mp4",
        )
        assert spec.duration_ms == 4_500 - 400
        assert_renders_to_claimed_length(spec)


class TestEffects:
    def test_slow_motion_lengthens_the_output(self, clips: dict[str, Path], workdir: Path) -> None:
        spec = spec_for(
            clips,
            (
                RenderSegment(
                    0, 0, 2_000, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.5),)
                ),
            ),
            workdir / "slow.mp4",
        )
        assert spec.duration_ms == 4_000
        assert_renders_to_claimed_length(spec)

    def test_speed_up_shortens_the_output(self, clips: dict[str, Path], workdir: Path) -> None:
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 4_000, effects=(Effect(kind=EffectKind.SPEED_UP, amount=2.0),)),),
            workdir / "fast.mp4",
        )
        assert spec.duration_ms == 2_000
        assert_renders_to_claimed_length(spec)

    def test_an_extreme_rate_still_renders(self, clips: dict[str, Path], workdir: Path) -> None:
        """0.25x is below what a single `atempo` accepts, so the audio chain has
        to factor it. The video side has no such limit, which is exactly the
        asymmetry that would go unnoticed without a test."""
        spec = spec_for(
            clips,
            (
                RenderSegment(
                    2, 0, 1_000, effects=(Effect(kind=EffectKind.SLOW_MOTION, amount=0.25),)
                ),
            ),
            workdir / "quarter.mp4",
            include_audio=True,
        )
        assert spec.duration_ms == 4_000
        assert_renders_to_claimed_length(spec)

    def test_speed_applies_to_audio_as_well(self, clips: dict[str, Path], workdir: Path) -> None:
        """A clip whose picture is slowed and whose sound is not would drift
        apart -- the failure a user notices immediately and a duration check
        does not."""
        spec = spec_for(
            clips,
            (RenderSegment(2, 0, 2_000, effects=(Effect(kind=EffectKind.SPEED_UP, amount=2.0),)),),
            workdir / "fast-audio.mp4",
            include_audio=True,
        )
        assert_renders_to_claimed_length(spec)

        streams = probe(Path(spec.output_path))["streams"]
        audio = next(s for s in streams if s["codec_type"] == "audio")
        video = next(s for s in streams if s["codec_type"] == "video")
        assert abs(float(audio["duration"]) - float(video["duration"])) < 0.2

    @pytest.mark.parametrize(
        ("kind", "amount"),
        [
            (EffectKind.BRIGHTNESS, 0.2),
            (EffectKind.CONTRAST, 1.3),
            (EffectKind.SATURATION, 1.6),
            (EffectKind.ZOOM_IN, 0.2),
            (EffectKind.ZOOM_OUT, 0.2),
        ],
    )
    def test_each_visual_effect_renders_without_changing_length(
        self, clips: dict[str, Path], workdir: Path, kind: EffectKind, amount: float
    ) -> None:
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 2_000, effects=(Effect(kind=kind, amount=amount),)),),
            workdir / f"{kind.value}.mp4",
        )
        assert spec.duration_ms == 2_000
        assert_renders_to_claimed_length(spec)

    def test_a_ranged_colour_effect_renders(self, clips: dict[str, Path], workdir: Path) -> None:
        spec = spec_for(
            clips,
            (
                RenderSegment(
                    0,
                    0,
                    3_000,
                    effects=(
                        Effect(kind=EffectKind.SATURATION, amount=1.8, start_ms=500, end_ms=2_000),
                    ),
                ),
            ),
            workdir / "ranged.mp4",
        )
        assert_renders_to_claimed_length(spec)

    def test_the_output_geometry_survives_a_zoom(
        self, clips: dict[str, Path], workdir: Path
    ) -> None:
        """`crop` changes the frame size; the chain has to scale it back."""
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 2_000, effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.3),)),),
            workdir / "zoom-geometry.mp4",
        )
        render(spec)
        video = next(
            s for s in probe(Path(spec.output_path))["streams"] if s["codec_type"] == "video"
        )
        assert (video["width"], video["height"]) == (320, 240)


class TestSubtitles:
    def track(self, **kwargs: object) -> SubtitleTrack:
        return SubtitleTrack(
            cues=(
                SubtitleCue(200, 1_600, "Hello world"),
                SubtitleCue(1_800, 3_400, "Xin chào thế giới — phụ đề tiếng Việt"),
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    def write_ass(self, workdir: Path, track: SubtitleTrack) -> str:
        path = workdir / "subtitles.ass"
        path.write_text(build_ass(track), encoding="utf-8")
        return str(path)

    def test_subtitles_burn_in(self, clips: dict[str, Path], workdir: Path) -> None:
        track = self.track()
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 2_000), RenderSegment(1, 0, 2_000)),
            workdir / "subs.mp4",
            subtitles=track,
        )
        assert_renders_to_claimed_length(spec, ass_path=self.write_ass(workdir, track))

    def test_burning_in_changes_the_picture(self, clips: dict[str, Path], workdir: Path) -> None:
        """The check that matters: subtitles are *visible*.

        A graph that parses and draws nothing would pass every other test here.
        Two renders of the same footage, one with subtitles and one without,
        must differ in the frames where a cue is on screen -- and the comparison
        is made against a frame where one is.
        """
        ffmpeg = _binary("ffmpeg")
        track = self.track()

        plain = spec_for(clips, (RenderSegment(0, 0, 3_000),), workdir / "plain.mp4")
        render(plain)

        burned = spec_for(
            clips, (RenderSegment(0, 0, 3_000),), workdir / "burned.mp4", subtitles=track
        )
        render(burned, ass_path=self.write_ass(workdir, track))

        frames: list[Path] = []
        for name, source in (("plain.png", plain), ("burned.png", burned)):
            frame = workdir / name
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    source.output_path,
                    "-ss",
                    "1",
                    "-frames:v",
                    "1",
                    str(frame),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            frames.append(frame)

        assert frames[0].read_bytes() != frames[1].read_bytes()
        # The burned frame carries more detail, so it compresses larger. A
        # subtitle that drew nothing would produce near-identical sizes.
        assert frames[1].stat().st_size > frames[0].stat().st_size

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_every_style_preset_renders(
        self, clips: dict[str, Path], workdir: Path, style: SubtitleStyle
    ) -> None:
        track = self.track(style=style)
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 3_500),),
            workdir / f"style-{style.value}.mp4",
            subtitles=track,
        )
        assert_renders_to_claimed_length(spec, ass_path=self.write_ass(workdir, track))

    @pytest.mark.parametrize("position", list(SubtitlePosition))
    def test_every_position_renders(
        self, clips: dict[str, Path], workdir: Path, position: SubtitlePosition
    ) -> None:
        track = self.track(position=position)
        spec = spec_for(
            clips,
            (RenderSegment(0, 0, 3_500),),
            workdir / f"pos-{position.value}.mp4",
            subtitles=track,
        )
        assert_renders_to_claimed_length(spec, ass_path=self.write_ass(workdir, track))

    def test_a_video_only_export_is_unaffected_by_the_subtitle_path(
        self, clips: dict[str, Path], workdir: Path
    ) -> None:
        """Passing no document must leave the graph exactly as it was."""
        spec = spec_for(clips, (RenderSegment(0, 0, 2_000),), workdir / "nosubs.mp4")
        argv = compile_render_argv(spec)
        assert "subtitles=" not in argv[argv.index("-filter_complex") + 1]
        assert "[vout]" in argv


class TestEverythingTogether:
    def test_transition_effect_and_subtitle_in_one_render(
        self, clips: dict[str, Path], workdir: Path
    ) -> None:
        """The case the acceptance exercises, at the compiler level.

        A crossfade, a colour grade, a zoom, a speed change and burned-in
        subtitles in one graph -- each of which alters either the timing or the
        frames the next one sees.
        """
        track = SubtitleTrack(
            cues=(SubtitleCue(300, 2_000, "Combined"), SubtitleCue(2_200, 3_500, "Tiếng Việt")),
            style=SubtitleStyle.BOLD,
            position=SubtitlePosition.BOTTOM,
        )
        ass = workdir / "combined.ass"
        ass.write_text(build_ass(track), encoding="utf-8")

        spec = spec_for(
            clips,
            (
                RenderSegment(
                    0, 0, 2_000, effects=(Effect(kind=EffectKind.BRIGHTNESS, amount=0.15),)
                ),
                RenderSegment(
                    1,
                    0,
                    2_000,
                    transition_in=TransitionKind.CROSSFADE,
                    transition_ms=500,
                    effects=(Effect(kind=EffectKind.ZOOM_IN, amount=0.15),),
                ),
                RenderSegment(2, 0, 2_000, effects=(Effect(kind=EffectKind.SPEED_UP, amount=2.0),)),
            ),
            workdir / "combined.mp4",
            subtitles=track,
            include_audio=True,
        )

        # 2000 + 2000 - 500 + 1000
        assert spec.duration_ms == 4_500
        assert_renders_to_claimed_length(spec, ass_path=str(ass))

        streams = probe(Path(spec.output_path))["streams"]
        assert any(s["codec_name"] == "h264" for s in streams)
        assert any(s["codec_name"] == "aac" for s in streams)

    def test_the_combined_output_decodes_completely(
        self, clips: dict[str, Path], workdir: Path
    ) -> None:
        track = SubtitleTrack(cues=(SubtitleCue(200, 1_500, "Decode me"),))
        ass = workdir / "decode.ass"
        ass.write_text(build_ass(track), encoding="utf-8")

        spec = spec_for(
            clips,
            (
                RenderSegment(0, 0, 2_000),
                RenderSegment(
                    1, 0, 2_000, transition_in=TransitionKind.CROSSFADE, transition_ms=400
                ),
            ),
            workdir / "decode.mp4",
            subtitles=track,
        )
        render(spec, ass_path=str(ass))

        decode = subprocess.run(
            [
                _binary("ffmpeg"),
                "-v",
                "error",
                "-xerror",
                "-i",
                spec.output_path,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=600,
        )
        assert decode.returncode == 0, decode.stderr.decode(errors="replace")[:800]
        assert not decode.stderr.strip()
