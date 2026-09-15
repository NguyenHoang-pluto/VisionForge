"""RenderSpec -> FFmpeg argv.

The only module in VisionForge that knows FFmpeg's command syntax. Everything
upstream deals in typed values; the translation to flags and filter strings
happens here and nowhere else.

Two properties make this safe and testable:

- **argv, never a shell string.** The result is a list. No concatenation, no
  quoting, no ``shell=True`` anywhere in the chain. A filename containing
  ``; rm -rf /`` is one argument, not two commands.
- **pure.** ``compile_render_argv`` touches no disk and spawns nothing, so the
  exact command produced for a given spec can be asserted in a unit test rather
  than inferred from whether a render happened to work.

The filter graph is built as a list of typed pieces and joined once at the end.
Building it by string concatenation scattered through the function is how filter
graphs become unmaintainable, and it is precisely the mistake this structure
exists to prevent.
"""

from __future__ import annotations

from visionforge.domain.editplan import FitMode
from visionforge.domain.timeline import RenderMusic, RenderSpec

#: One sample rate and layout for everything that meets in the mixer. `concat`
#: and `amix` both require their inputs to agree, and real sources do not: a
#: 44.1 kHz mono voice memo and a 48 kHz stereo track have to be made
#: interchangeable before either filter will touch them.
MIX_SAMPLE_RATE = 48_000
MIX_CHANNEL_LAYOUT = "stereo"
_AFORMAT = f"aformat=sample_rates={MIX_SAMPLE_RATE}:channel_layouts={MIX_CHANNEL_LAYOUT}"

#: ``-progress pipe:1`` emits key=value lines we parse for real progress.
#: Without it the only signal is stderr text, which changes between versions.
PROGRESS_ARGS: tuple[str, ...] = ("-progress", "pipe:1", "-nostats")


def _ms(value: int) -> str:
    """Milliseconds as seconds with millisecond precision.

    FFmpeg accepts seconds; formatting explicitly avoids scientific notation on
    small values and locale-dependent separators.
    """
    return f"{value / 1000:.3f}"


def scale_filter(width: int, height: int, fit: FitMode) -> str:
    """Fit a source frame into the output rectangle.

    ``cover`` scales so the frame fills the rectangle and crops the overflow;
    ``contain`` scales so it fits entirely and pads the remainder. Both then
    ``setsar=1`` -- without it a source with a non-square sample aspect ratio
    produces a correctly-sized but horizontally stretched output, which is a
    subtle and very confusing bug.
    """
    if fit is FitMode.COVER:
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def _gain(value: float) -> str:
    """A linear gain, formatted.

    Rounded to four places rather than passed through: a float repr like
    ``0.30000000000000004`` is a filter argument that differs between platforms,
    and the command has to be byte-identical for a given spec or the unit tests
    are asserting nothing.
    """
    return f"{round(value, 4):g}"


def build_music_chain(music: RenderMusic, *, label: str) -> str:
    """Trim, level, shape and position the music bed.

    Order is not arbitrary and is the whole of what makes the result correct:

    ``atrim`` then ``asetpts``  take the requested passage and rebase it to zero,
                                for the same reason the video trim does -- the
                                downstream filters measure from the start of
                                *this* stream, not from the start of the file.
    ``aformat``                 make it mixable before anything else touches it.
    ``volume``                  level before the fades, so a fade ramps to the
                                level the user chose rather than to unity and
                                then down.
    ``afade`` in / out          the envelope, with the out point measured from
                                the start of the trimmed passage.
    ``adelay``                  position on the output timeline. Last, because
                                it prepends silence, and every offset above
                                would otherwise have to account for it.
    """
    parts = [
        f"[{music.input_index}:a]",
        f"atrim=start={_ms(music.source_in_ms)}:end={_ms(music.source_out_ms)}",
        "asetpts=PTS-STARTPTS",
        _AFORMAT,
    ]
    if music.gain != 1.0:
        parts.append(f"volume={_gain(music.gain)}")
    if music.fade_in_ms > 0:
        parts.append(f"afade=t=in:st=0:d={_ms(music.fade_in_ms)}")
    if music.fade_out_ms > 0:
        start = max(0, music.duration_ms - music.fade_out_ms)
        parts.append(f"afade=t=out:st={_ms(start)}:d={_ms(music.fade_out_ms)}")
    if music.timeline_start_ms > 0:
        # `all=1` applies the delay to every channel. Without it only the first
        # is delayed and the bed arrives with its stereo image torn in half.
        parts.append(f"adelay={music.timeline_start_ms}:all=1")

    head, *rest = parts
    return f"{head}{','.join(rest)}[{label}]"


def build_audio_graph(spec: RenderSpec, source_labels: list[str]) -> list[str]:
    """The audio side of the graph, ending at ``[aout]``.

    Four shapes, and the last line is the same in all of them: whatever the
    audio is, it is cut to exactly the video's length. That guarantee lives here
    rather than being left to ``-shortest`` because ``-shortest`` decides by
    whichever stream ends first, which is the right answer only by luck.
    """
    if not spec.has_audio_output:
        return []

    parts: list[str] = []
    mix_inputs: list[str] = []

    if spec.include_audio and source_labels:
        chain = "".join(f"[{label}]" for label in source_labels)
        parts.append(f"{chain}concat=n={len(source_labels)}:v=0:a=1[asrc]")
        if spec.source_gain != 1.0:
            parts.append(f"[asrc]volume={_gain(spec.source_gain)}[asrcg]")
            mix_inputs.append("asrcg")
        else:
            mix_inputs.append("asrc")

    if spec.music is not None:
        parts.append(build_music_chain(spec.music, label="amus"))
        mix_inputs.append("amus")

    if len(mix_inputs) == 2:
        chain = "".join(f"[{label}]" for label in mix_inputs)
        # `normalize=0` keeps each input at the level the plan asked for. With
        # normalisation on, amix divides by the input count, so adding a music
        # bed would silently halve the dialogue -- a gain the user never set and
        # cannot see in the plan.
        parts.append(
            f"{chain}amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[amix]"
        )
        mixed = "amix"
    else:
        mixed = mix_inputs[0]

    # Pad then trim: pad covers music shorter than the picture, trim covers
    # music longer than it. Together they make the audio exactly as long as the
    # video regardless of which was longer, with no reliance on -shortest.
    parts.append(
        f"[{mixed}]apad,atrim=end={_ms(spec.duration_ms)},asetpts=PTS-STARTPTS,{_AFORMAT}[aout]"
    )
    return parts


def build_filter_graph(spec: RenderSpec) -> str:
    """The complete filtergraph: trim, normalise, concatenate.

    Each segment is trimmed from its input, reset to a zero timebase, scaled to
    the output rectangle and forced to the output frame rate. Normalising
    *before* the concat is required, not stylistic: ``concat`` demands that every
    input share dimensions, pixel format and frame rate, and sources in a real
    folder will not.
    """
    parts: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    for index, segment in enumerate(spec.segments):
        source = segment.input_index
        video_label = f"v{index}"
        parts.append(
            f"[{source}:v]"
            f"trim=start={_ms(segment.source_in_ms)}:end={_ms(segment.source_out_ms)},"
            # setpts rebases each trimmed piece to start at zero; without it the
            # concatenated output inherits the source timestamps and stalls.
            f"setpts=PTS-STARTPTS,"
            f"{scale_filter(spec.width, spec.height, spec.fit)},"
            f"fps={spec.fps},"
            f"format={spec.pixel_format}"
            f"[{video_label}]"
        )
        video_labels.append(video_label)

        if spec.include_audio:
            audio_label = f"a{index}"
            parts.append(
                f"[{source}:a]"
                f"atrim=start={_ms(segment.source_in_ms)}:end={_ms(segment.source_out_ms)},"
                f"asetpts=PTS-STARTPTS,"
                # Resample to one rate/layout for the same reason as the video:
                # concat requires uniform inputs.
                f"{_AFORMAT}"
                f"[{audio_label}]"
            )
            audio_labels.append(audio_label)

    # Video is concatenated on its own, and the clips' audio separately.
    #
    # Phase 4 concatenated both in one `concat=v=1:a=1`, which was correct while
    # audio could only ever follow the cuts. It cannot survive a music bed: the
    # mix has to happen after the concat, and a single filter that emits both
    # streams leaves nowhere to put it. Splitting them changes no output when
    # there is no music -- the same segments, in the same order, at the same
    # rate -- and makes the mix expressible when there is.
    count = len(spec.segments)
    chain = "".join(f"[{v}]" for v in video_labels)
    parts.append(f"{chain}concat=n={count}:v=1:a=0[vout]")
    parts.extend(build_audio_graph(spec, audio_labels))

    return ";".join(parts)


def compile_render_argv(spec: RenderSpec, *, overwrite: bool = True) -> list[str]:
    """Compile a RenderSpec into FFmpeg arguments.

    Returns the arguments *after* the executable: the runner resolves and
    prepends the binary, so this function never has to know where FFmpeg lives
    and cannot be pointed at a different one.

    Pure. Given a spec, the output is fully determined -- which is what lets the
    command be asserted directly in tests instead of being inferred from whether
    a render succeeded.
    """
    if not spec.inputs:
        raise ValueError("render spec has no inputs")
    if not spec.segments:
        raise ValueError("render spec has no segments")

    args: list[str] = ["-hide_banner", "-loglevel", "error", *PROGRESS_ARGS]
    if overwrite:
        args.append("-y")

    for render_input in spec.inputs:
        args += ["-i", render_input.local_path]

    args += ["-filter_complex", build_filter_graph(spec)]
    args += ["-map", "[vout]"]
    if spec.has_audio_output:
        args += ["-map", "[aout]"]

    args += [
        "-c:v",
        "libx264",
        "-preset",
        spec.preset,
        "-crf",
        str(spec.crf),
        "-pix_fmt",
        spec.pixel_format,
        # Closed GOP at one second keeps the file seekable in a browser without
        # the size cost of a shorter interval.
        "-g",
        str(spec.fps * 2),
        "-r",
        str(spec.fps),
    ]

    if spec.has_audio_output:
        args += [
            "-c:a",
            "aac",
            "-b:a",
            f"{spec.audio_bitrate_kbps}k",
            "-ar",
            str(MIX_SAMPLE_RATE),
        ]
    else:
        args.append("-an")

    args += [
        # Moves the index to the front so playback can start before the whole
        # file has downloaded. Costs one extra pass over the output.
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        spec.output_path,
    ]
    return args


def parse_progress_line(line: str) -> tuple[str, str] | None:
    """One ``key=value`` line from ``-progress pipe:1``.

    Returns ``None`` for anything that is not a recognisable pair, so a malformed
    line cannot break progress reporting.
    """
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if not key or not value:
        return None
    return key, value


def progress_fraction(out_time_us: int, total_ms: int) -> float:
    """Fraction complete from FFmpeg's ``out_time_us`` against known duration.

    Clamped: FFmpeg occasionally reports a timestamp slightly past the end, and
    a progress bar that reads 103% is worse than one that stops at 100%.
    """
    if total_ms <= 0:
        return 0.0
    return max(0.0, min(1.0, (out_time_us / 1000.0) / total_ms))


__all__ = [
    "MIX_CHANNEL_LAYOUT",
    "MIX_SAMPLE_RATE",
    "PROGRESS_ARGS",
    "build_audio_graph",
    "build_filter_graph",
    "build_music_chain",
    "compile_render_argv",
    "parse_progress_line",
    "progress_fraction",
    "scale_filter",
]
