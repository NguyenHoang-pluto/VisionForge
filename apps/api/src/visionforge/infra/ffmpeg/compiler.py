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

from visionforge.domain.editplan import FitMode, TransitionKind
from visionforge.domain.effects import EffectKind
from visionforge.domain.timeline import RenderMusic, RenderSegment, RenderSpec
from visionforge.infra.ffmpeg.subtitles import escape_filter_path

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


def build_effect_chain(segment: RenderSegment, *, fps: int, width: int, height: int) -> str:
    """The filters one segment's effects compile to, in a fixed order.

    Order is decided here rather than by the caller's list, because it changes
    the picture: a colour grade applied before a zoom is sampled differently
    from one applied after, and "whichever order the client happened to send"
    is not a rendering contract. Geometry first, then colour, then rate.

    Returns a chain ending in a comma when non-empty, so callers can splice it
    into a filter string without counting separators.
    """
    parts: list[str] = []
    duration_ms = segment.source_duration_ms

    # --- geometry ---
    for effect in segment.effects:
        if effect.kind not in (EffectKind.ZOOM_IN, EffectKind.ZOOM_OUT) or effect.is_neutral:
            continue
        # A centre crop that tightens (or loosens) over the clip, scaled back
        # out to the output rectangle. `crop` evaluates its expressions once per
        # frame and leaves timestamps alone, which is the whole reason it is
        # used here instead of `zoompan`: zoompan regenerates PTS from its own
        # frame counter, and a two-second clip came back claiming seventeen
        # minutes.
        #
        # `t` is the segment's own clock -- setpts rebased it to zero above --
        # so the ramp is written against the trim's length and needs no frame
        # count.
        seconds = max(duration_ms / 1000, 0.001)
        amount = effect.amount
        if effect.kind is EffectKind.ZOOM_IN:
            factor = rf"1+{amount:g}*min(t/{seconds:g}\,1)"
        else:
            factor = rf"{1 + amount:g}-{amount:g}*min(t/{seconds:g}\,1)"
        parts.append(
            f"crop=w='iw/({factor})':h='ih/({factor})'"
            f":x='(iw-ow)/2':y='(ih-oh)/2'"
            f",scale={width}:{height},setsar=1"
        )

    # --- colour ---
    #
    # One `eq` per effect rather than one combined: they may have different
    # windows, and `enable` applies to a filter instance.
    for effect in segment.effects:
        option = {
            EffectKind.BRIGHTNESS: "brightness",
            EffectKind.CONTRAST: "contrast",
            EffectKind.SATURATION: "saturation",
        }.get(effect.kind)
        if option is None or effect.is_neutral:
            continue
        chain = f"eq={option}={effect.amount:g}"
        if effect.is_ranged:
            start, end = effect.window(duration_ms)
            # Timeline-gated. `t` here is the trimmed segment's own clock,
            # because setpts has already rebased it to zero.
            chain += f":enable='between(t,{_ms(start)},{_ms(end)})'"
        parts.append(chain)

    # --- rate ---
    #
    # Last, and deliberately so: setpts rewrites timestamps, and a filter that
    # reasons about time (the `enable` above) must see the unaltered clock.
    rate = segment.speed
    if abs(rate - 1.0) > 1e-9:
        parts.append(f"setpts=PTS/{rate:g}")

    return ",".join(parts) + "," if parts else ""


def build_audio_effect_chain(segment: RenderSegment) -> str:
    """The audio side of a speed change.

    ``atempo`` is the only effect audio has, because the others are visual. Its
    accepted range is 0.5-2.0 on the builds this has to work with, so a rate
    outside that is factored into a chain of instances inside it -- 0.25 becomes
    two halvings. Colour and zoom produce nothing here.
    """
    rate = segment.speed
    if abs(rate - 1.0) <= 1e-9:
        return ""

    factors: list[float] = []
    remaining = rate
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)

    return ",".join(f"atempo={factor:g}" for factor in factors) + ","


def build_subtitle_filter(spec: RenderSpec, ass_path: str) -> str:
    """Burn subtitles in, from a document this process wrote.

    The filename is the only thing that crosses into the graph, and it names a
    file in the render's own scratch directory. No user string reaches here:
    the text is inside the document, which libass parses as text.
    """
    return f"subtitles=filename='{escape_filter_path(ass_path)}'"


def build_filter_graph(spec: RenderSpec, *, ass_path: str | None = None) -> str:
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
            # Effects sit after normalisation and before the frame rate is
            # forced: zoompan needs to know the geometry it is cropping within,
            # and a speed change has to be reflected in the timestamps that fps
            # then resamples.
            f"{build_effect_chain(segment, fps=spec.fps, width=spec.width, height=spec.height)}"
            f"{_fade_chain(segment, spec.fps)}"
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
                f"{build_audio_effect_chain(segment)}"
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
    if spec.has_transitions:
        # Pairwise, because `concat` cannot overlap. See `_join_with_transitions`.
        parts.extend(_join_with_transitions(spec, video_labels, "vout"))
    else:
        # The Phase 4 path, untouched. Every plan written before Phase 9 takes
        # this branch and produces the bytes it always did.
        chain = "".join(f"[{v}]" for v in video_labels)
        parts.append(f"{chain}concat=n={count}:v=1:a=0[vout]")

    parts.extend(build_audio_graph(spec, audio_labels))

    if ass_path is not None and spec.subtitles is not None:
        # Burned in last, over the finished picture, so a cue that spans a cut
        # is drawn once rather than clipped at the join -- and so subtitles are
        # not dissolved by a crossfade they happen to overlap.
        parts.append(f"[vout]{build_subtitle_filter(spec, ass_path)}[vsub]")

    return ";".join(parts)


def _fade_chain(segment: RenderSegment, fps: int) -> str:
    """Fades to and from black, which are drawn on the clip rather than between.

    Unlike a crossfade these consume no time: the clip plays its whole length
    and the fade is painted over its head or tail. That is why they are here, in
    the per-segment chain, and not in the join.
    """
    if segment.transition_ms <= 0:
        return ""

    duration_s = segment.transition_ms / 1000
    if segment.transition_in is TransitionKind.FADE_IN:
        return f"fade=t=in:st=0:d={duration_s:g},"
    if segment.transition_in is TransitionKind.FADE_TO_BLACK:
        # Measured from the segment's *output* length, so a slowed clip fades
        # out at its real end rather than early.
        start = max(0.0, segment.duration_ms / 1000 - duration_s)
        return f"fade=t=out:st={start:g}:d={duration_s:g},"
    return ""


def _join_with_transitions(spec: RenderSpec, labels: list[str], output_label: str) -> list[str]:
    """Join segments pairwise, dissolving where asked.

    `concat` takes n inputs and plays them end to end; it has no way to overlap
    two of them, so a crossfade cannot be expressed in the same filter. `xfade`
    can, but takes exactly two inputs -- so the chain is built one join at a
    time, carrying a running label and a running output length.

    The offset arithmetic is the part worth being careful about. `xfade`'s
    ``offset`` is measured from the start of the *accumulated* first input, and
    the transition begins there, so joining an accumulation of length ``A`` to a
    clip of length ``B`` with a ``D``-long dissolve puts the offset at ``A - D``
    and produces ``A + B - D``. Getting that wrong does not fail: it renders,
    with the dissolve in the wrong place and the output the wrong length.

    Cuts inside a transitioned spec become two-input concats rather than being
    batched, which is exactly equivalent and keeps one code path.
    """
    # Every join re-states the timebase.
    #
    # `concat` emits 1/1000000 whatever its inputs were, while a segment chain
    # ending in `fps=` emits 1/fps -- and `xfade` refuses two inputs whose
    # timebases differ. That only bites when a cut precedes a dissolve, which is
    # why it survived the first round of testing and was caught by the case that
    # mixes the two.
    timebase = f"settb=1/{spec.fps}"

    parts: list[str] = []
    current = labels[0]
    accumulated = spec.segments[0].duration_ms

    for index in range(1, len(spec.segments)):
        segment = spec.segments[index]
        nxt = labels[index]
        last = index == len(spec.segments) - 1
        out = output_label if last else f"x{index}"

        if segment.transition_in is TransitionKind.CROSSFADE and segment.transition_ms > 0:
            offset_s = max(0.0, (accumulated - segment.transition_ms) / 1000)
            parts.append(
                f"[{current}][{nxt}]"
                f"xfade=transition=fade:duration={segment.transition_ms / 1000:g}"
                f":offset={offset_s:g}"
                f",{timebase}"
                f"[{out}]"
            )
            accumulated = accumulated + segment.duration_ms - segment.transition_ms
        else:
            parts.append(f"[{current}][{nxt}]concat=n=2:v=1:a=0,{timebase}[{out}]")
            accumulated += segment.duration_ms

        current = out

    return parts


def compile_render_argv(
    spec: RenderSpec, *, overwrite: bool = True, ass_path: str | None = None
) -> list[str]:
    """Compile a RenderSpec into FFmpeg arguments.

    Returns the arguments *after* the executable: the runner resolves and
    prepends the binary, so this function never has to know where FFmpeg lives
    and cannot be pointed at a different one.

    ``ass_path`` names a subtitle document the *worker* has already written into
    the render's scratch directory. It is an argument rather than something this
    function derives, because writing a file is I/O and this function is pure --
    which is what lets the whole command be asserted in a test instead of being
    inferred from whether a render succeeded.
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

    burn_subtitles = ass_path is not None and spec.subtitles is not None
    args += ["-filter_complex", build_filter_graph(spec, ass_path=ass_path)]
    args += ["-map", "[vsub]" if burn_subtitles else "[vout]"]
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
