"""The audio half of the filter graph.

``compile_render_argv`` is pure, so every assertion here is against the exact
command rather than against whether a render happened to work. That is the
property that makes a filter graph reviewable: the trim, the level, the
envelope and the mix policy are all visible in a string a test can pin.

Two things are being defended. The first is correctness -- gain before fades,
delay last, audio cut to the video's length. The second is the security posture:
there is no path by which a plan can put a token into this command, and the
tests that prove it are at the bottom.
"""

from __future__ import annotations

import re
import uuid

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
from visionforge.domain.timeline import (
    RenderInput,
    RenderMusic,
    RenderSegment,
    RenderSpec,
    build_render_spec,
    compile_timeline,
)
from visionforge.infra.ffmpeg.compiler import (
    MIX_SAMPLE_RATE,
    build_audio_graph,
    build_filter_graph,
    build_music_chain,
    compile_render_argv,
)

PROJECT = ProjectId(uuid.uuid4())
CLIP = MediaId(uuid.uuid4())
TRACK = MediaId(uuid.uuid4())


def music(**overrides: object) -> RenderMusic:
    values: dict = {
        "input_index": 1,
        "source_in_ms": 0,
        "source_out_ms": 10_000,
        "timeline_start_ms": 0,
        "gain": 1.0,
        "fade_in_ms": 0,
        "fade_out_ms": 0,
    }
    values.update(overrides)
    return RenderMusic(**values)  # type: ignore[arg-type]


def spec(
    *,
    segments: int = 2,
    include_audio: bool = False,
    source_gain: float = 1.0,
    with_music: RenderMusic | None = None,
) -> RenderSpec:
    inputs = [RenderInput(media_id=CLIP, index=0, local_path="/w/clip.mp4")]
    if with_music is not None:
        inputs.append(RenderInput(media_id=TRACK, index=1, local_path="/w/track.mp3"))
    return RenderSpec(
        inputs=tuple(inputs),
        segments=tuple(
            RenderSegment(input_index=0, source_in_ms=i * 3_000, source_out_ms=(i + 1) * 3_000)
            for i in range(segments)
        ),
        width=1280,
        height=720,
        fps=30,
        fit=FitMode.COVER,
        output_path="/w/out.mp4",
        include_audio=include_audio,
        source_gain=source_gain,
        music=with_music,
    )


def stage(graph: str, label: str) -> str:
    """The one chain in the graph that ends at ``[label]``."""
    matches = [part for part in graph.split(";") if part.endswith(f"[{label}]")]
    assert len(matches) == 1, f"expected exactly one chain ending in [{label}], got {matches}"
    return matches[0]


# ------------------------------------------------------------------ no audio
class TestSilentOutput:
    def test_a_spec_with_neither_source_nor_music_has_no_audio_output(self) -> None:
        assert spec().has_audio_output is False

    def test_the_graph_has_no_audio_chain(self) -> None:
        assert build_audio_graph(spec(), []) == []

    def test_the_command_disables_audio_explicitly(self) -> None:
        args = compile_render_argv(spec())
        assert "-an" in args
        assert "-c:a" not in args

    def test_nothing_is_mapped_to_aout(self) -> None:
        assert "[aout]" not in compile_render_argv(spec())


# -------------------------------------------------------------- source only
class TestSourceAudioOnly:
    def test_source_audio_is_concatenated_then_cut_to_length(self) -> None:
        graph = build_filter_graph(spec(include_audio=True))
        assert "concat=n=2:v=0:a=1[asrc]" in graph
        assert "atrim=end=6.000" in stage(graph, "aout")

    def test_unity_gain_emits_no_volume_filter(self) -> None:
        """A filter that does nothing is a filter that can be wrong."""
        graph = build_filter_graph(spec(include_audio=True, source_gain=1.0))
        assert "volume=" not in graph

    def test_a_non_unity_gain_is_applied_after_the_concat(self) -> None:
        graph = build_filter_graph(spec(include_audio=True, source_gain=0.25))
        assert "[asrc]volume=0.25[asrcg]" in graph
        assert stage(graph, "aout").startswith("[asrcg]")

    def test_the_encoder_is_configured_for_audio(self) -> None:
        args = compile_render_argv(spec(include_audio=True))
        assert "-an" not in args
        assert args[args.index("-c:a") + 1] == "aac"
        assert args[args.index("-ar") + 1] == str(MIX_SAMPLE_RATE)


# --------------------------------------------------------------- music chain
class TestMusicChain:
    def test_the_bed_is_trimmed_and_rebased(self) -> None:
        chain = build_music_chain(music(source_in_ms=2_000, source_out_ms=9_000), label="amus")
        assert "atrim=start=2.000:end=9.000" in chain
        assert "asetpts=PTS-STARTPTS" in chain

    def test_it_is_made_mixable_before_anything_else_touches_it(self) -> None:
        chain = build_music_chain(music(gain=0.5), label="amus")
        assert chain.index("aformat=") < chain.index("volume=")

    def test_gain_is_applied_before_the_fades(self) -> None:
        """Otherwise a fade ramps to unity and is then scaled, which is audible."""
        chain = build_music_chain(music(gain=0.4, fade_in_ms=1_000), label="amus")
        assert chain.index("volume=") < chain.index("afade=t=in")

    def test_the_fade_out_is_measured_from_the_end_of_the_trimmed_passage(self) -> None:
        chain = build_music_chain(
            music(source_in_ms=5_000, source_out_ms=15_000, fade_out_ms=2_000), label="amus"
        )
        # 10 s of audio, fading out over the last 2 s: starts at 8 s.
        assert "afade=t=out:st=8.000:d=2.000" in chain

    def test_a_zero_fade_emits_no_filter(self) -> None:
        chain = build_music_chain(music(fade_in_ms=0, fade_out_ms=0), label="amus")
        assert "afade" not in chain

    def test_the_delay_is_last_and_applies_to_every_channel(self) -> None:
        """adelay prepends silence; every offset above it is measured before it."""
        chain = build_music_chain(music(timeline_start_ms=4_000, fade_in_ms=500), label="amus")
        assert chain.index("afade") < chain.index("adelay")
        assert "adelay=4000:all=1" in chain

    def test_no_delay_at_the_origin(self) -> None:
        assert "adelay" not in build_music_chain(music(timeline_start_ms=0), label="amus")

    def test_the_chain_reads_from_its_own_input_index(self) -> None:
        assert build_music_chain(music(input_index=3), label="amus").startswith("[3:a]")

    def test_gain_is_formatted_stably(self) -> None:
        """A float repr that differs by platform is a command that differs by
        platform, and then these tests assert nothing."""
        chain = build_music_chain(music(gain=0.1 + 0.2), label="amus")
        assert "volume=0.3" in chain


# --------------------------------------------------------------- music only
class TestMusicOnly:
    def test_music_alone_produces_an_audio_stream(self) -> None:
        s = spec(include_audio=False, with_music=music())
        assert s.has_audio_output is True
        assert "-an" not in compile_render_argv(s)

    def test_there_is_no_mix_with_a_single_source(self) -> None:
        graph = build_filter_graph(spec(with_music=music()))
        assert "amix" not in graph

    def test_the_clips_own_audio_is_not_read(self) -> None:
        """Source audio off means the video inputs' audio is never opened."""
        graph = build_filter_graph(spec(with_music=music()))
        assert "[0:a]" not in graph

    def test_music_longer_than_the_picture_is_trimmed_to_it(self) -> None:
        graph = build_filter_graph(spec(segments=2, with_music=music(source_out_ms=60_000)))
        assert "atrim=end=6.000" in stage(graph, "aout")

    def test_music_shorter_than_the_picture_is_padded_to_it(self) -> None:
        """apad then atrim: the pair makes the audio exactly the video's length
        whichever was longer, without relying on -shortest."""
        graph = build_filter_graph(spec(segments=4, with_music=music(source_out_ms=2_000)))
        final = stage(graph, "aout")
        assert final.index("apad") < final.index("atrim=end=12.000")


# ---------------------------------------------------------------- the mix
class TestMix:
    def test_both_sources_are_mixed(self) -> None:
        graph = build_filter_graph(spec(include_audio=True, with_music=music()))
        assert "[asrc][amus]amix=inputs=2" in graph

    def test_the_mix_does_not_normalise(self) -> None:
        """With normalisation on, amix halves every input -- so adding a bed
        would silently duck the dialogue by a gain nobody set."""
        graph = build_filter_graph(spec(include_audio=True, with_music=music()))
        assert "normalize=0" in graph

    def test_the_mix_takes_the_longest_input(self) -> None:
        graph = build_filter_graph(spec(include_audio=True, with_music=music()))
        assert "duration=longest" in graph

    def test_independent_gains_reach_the_mix_separately(self) -> None:
        graph = build_filter_graph(
            spec(include_audio=True, source_gain=0.2, with_music=music(gain=0.9))
        )
        assert "[asrc]volume=0.2[asrcg]" in graph
        assert "volume=0.9" in stage(graph, "amus")
        assert "[asrcg][amus]amix" in graph

    def test_the_mixed_result_is_cut_to_the_video_length(self) -> None:
        graph = build_filter_graph(
            spec(segments=3, include_audio=True, with_music=music(source_out_ms=60_000))
        )
        assert stage(graph, "aout").startswith("[amix]")
        assert "atrim=end=9.000" in stage(graph, "aout")

    def test_exactly_one_chain_produces_aout(self) -> None:
        for s in (
            spec(include_audio=True),
            spec(with_music=music()),
            spec(include_audio=True, with_music=music()),
        ):
            stage(build_filter_graph(s), "aout")  # raises unless exactly one


# ------------------------------------------------------- timeline integration
class TestFromTheTimeline:
    def _plan(self, cue: MusicCue | None, audio: AudioMode = AudioMode.NONE) -> EditPlan:
        return EditPlan(
            project_id=PROJECT,
            segments=(
                Segment(media_id=CLIP, order=0, source_in_ms=0, source_out_ms=4_000),
                Segment(media_id=CLIP, order=1, source_in_ms=4_000, source_out_ms=8_000),
            ),
            output=OutputSpec(aspect_ratio=AspectRatio.LANDSCAPE_16_9, audio=audio),
            music=cue,
        )

    def test_a_plan_with_music_compiles_all_the_way_to_argv(self) -> None:
        cue = MusicCue(
            media_id=TRACK,
            source_in_ms=0,
            source_out_ms=30_000,
            gain=0.5,
            fade_in_ms=1_000,
            fade_out_ms=1_500,
        )
        timeline = compile_timeline(self._plan(cue, AudioMode.SOURCE))
        built = build_render_spec(
            timeline,
            local_paths={CLIP: "/w/clip.mp4", TRACK: "/w/track.mp3"},
            output_path="/w/out.mp4",
        )
        args = compile_render_argv(built)

        assert built.music is not None
        # Two segments of one clip are two inputs (Phase 12), plus the track.
        assert args.count("-i") == 3
        assert "-map" in args and "[aout]" in args

    def test_the_music_file_becomes_an_ordinary_input(self) -> None:
        cue = MusicCue(media_id=TRACK, source_in_ms=0, source_out_ms=30_000)
        timeline = compile_timeline(self._plan(cue))
        built = build_render_spec(
            timeline,
            local_paths={CLIP: "/w/clip.mp4", TRACK: "/w/track.mp3"},
            output_path="/w/out.mp4",
        )
        assert [i.local_path for i in built.inputs] == [
            "/w/clip.mp4",
            "/w/clip.mp4",
            "/w/track.mp3",
        ]
        assert built.music is not None
        assert built.music.input_index == 2

    def test_a_missing_music_path_is_refused(self) -> None:
        cue = MusicCue(media_id=TRACK, source_in_ms=0, source_out_ms=30_000)
        timeline = compile_timeline(self._plan(cue))
        with pytest.raises(ValueError, match="no local path"):
            build_render_spec(timeline, local_paths={CLIP: "/w/clip.mp4"}, output_path="/w/out.mp4")

    def test_a_plan_without_music_produces_the_phase_six_spec(self) -> None:
        timeline = compile_timeline(self._plan(None, AudioMode.SOURCE))
        built = build_render_spec(
            timeline, local_paths={CLIP: "/w/clip.mp4"}, output_path="/w/out.mp4"
        )
        assert built.music is None
        assert built.source_gain == 1.0
        assert len(built.inputs) == 2


# ------------------------------------------------------------------ security
class TestNoInjectionSurface:
    """The audio path gets the same guarantee the video path has.

    An audio filter graph can run commands as readily as a video one, so the
    argument has to hold here too: every value that reaches the command comes
    from a number or a server-resolved path, and there is no field in the plan
    through which a caller could add a token.
    """

    def test_the_command_is_argv_not_a_string(self) -> None:
        args = compile_render_argv(spec(include_audio=True, with_music=music()))
        assert isinstance(args, list)
        assert all(isinstance(a, str) for a in args)

    def test_the_whole_filter_graph_is_one_argument(self) -> None:
        args = compile_render_argv(spec(include_audio=True, with_music=music()))
        graph = args[args.index("-filter_complex") + 1]
        assert graph.count("concat=") >= 1
        assert args.count("-filter_complex") == 1

    def test_a_hostile_filename_stays_one_argument(self) -> None:
        hostile = "/w/track; rm -rf /.mp3"
        built = RenderSpec(
            inputs=(
                RenderInput(media_id=CLIP, index=0, local_path="/w/clip.mp4"),
                RenderInput(media_id=TRACK, index=1, local_path=hostile),
            ),
            segments=(RenderSegment(input_index=0, source_in_ms=0, source_out_ms=3_000),),
            width=1280,
            height=720,
            fps=30,
            fit=FitMode.COVER,
            output_path="/w/out.mp4",
            music=music(),
        )
        args = compile_render_argv(built)
        assert hostile in args
        assert args.count(hostile) == 1

    #: Every word the audio graph is allowed to contain: filter names, their
    #: option names, the handful of keyword values, and the stream labels this
    #: module generates. Pinned as a set so that a free string reaching the
    #: graph from a plan fails here rather than being discovered in an encode.
    AUDIO_VOCABULARY = frozenset(
        {
            # filters
            "concat",
            "volume",
            "atrim",
            "asetpts",
            "aformat",
            "afade",
            "adelay",
            "apad",
            "amix",
            # option names
            "n",
            "v",
            "a",
            "start",
            "end",
            "sample_rates",
            "channel_layouts",
            "t",
            "st",
            "d",
            "all",
            "inputs",
            "duration",
            "dropout_transition",
            "normalize",
            # keyword values
            "in",
            "out",
            "stereo",
            "longest",
            "PTS-STARTPTS",
            # stream labels
            "a0",
            "a1",
            "asrc",
            "asrcg",
            "amus",
            "aout",
        }
    )

    def test_the_audio_graph_contains_no_word_this_module_did_not_write(self) -> None:
        """The property behind the field list: nothing user-supplied is a word.

        Gains, fades, offsets and trims are numbers by the time they reach here.
        So every alphabetic token in the graph is one this module chose, and
        every other token is a number -- which together mean there is nowhere
        for a caller-supplied string to appear, whatever the plan said.
        """
        graph = ";".join(
            build_audio_graph(
                spec(
                    include_audio=True,
                    source_gain=0.35,
                    with_music=music(
                        gain=0.8, fade_in_ms=500, fade_out_ms=900, timeline_start_ms=1_200
                    ),
                ),
                ["a0", "a1"],
            )
        )

        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_\-]*", graph))
        assert words <= self.AUDIO_VOCABULARY, f"unexpected words: {words - self.AUDIO_VOCABULARY}"

        numbers = re.split(r"[A-Za-z_][A-Za-z0-9_\-]*|[\[\]=,;:]", graph)
        for token in numbers:
            assert token == "" or re.fullmatch(
                r"[0-9.]+", token
            ), f"{token!r} is neither a number nor a word this module chose"
