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
from visionforge.domain.timeline import RenderSpec

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
                f"aformat=sample_rates=48000:channel_layouts=stereo"
                f"[{audio_label}]"
            )
            audio_labels.append(audio_label)

    count = len(spec.segments)
    if spec.include_audio:
        chain = "".join(f"[{v}][{a}]" for v, a in zip(video_labels, audio_labels, strict=True))
        parts.append(f"{chain}concat=n={count}:v=1:a=1[vout][aout]")
    else:
        chain = "".join(f"[{v}]" for v in video_labels)
        parts.append(f"{chain}concat=n={count}:v=1:a=0[vout]")

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
    if spec.include_audio:
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

    if spec.include_audio:
        args += ["-c:a", "aac", "-b:a", f"{spec.audio_bitrate_kbps}k", "-ar", "48000"]
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
    "PROGRESS_ARGS",
    "build_filter_graph",
    "compile_render_argv",
    "parse_progress_line",
    "progress_fraction",
    "scale_filter",
]
