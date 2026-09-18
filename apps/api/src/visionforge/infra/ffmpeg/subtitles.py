"""Turning a subtitle track into an ASS document.

This module exists so that user text never becomes part of a filter graph.

FFmpeg's ``drawtext`` is the obvious way to burn in subtitles and the wrong one:
the text goes *into the filter string*, where ``:`` separates options, ``\\``
escapes, ``'`` quotes and ``%`` expands. A subtitle reading ``12:30`` is then a
syntax error, and one reading ``':drawbox=c=red@1:t=fill,drawtext=text='`` is an
arbitrary filter chain supplied by whoever typed the subtitle.

ASS moves the text off the command line entirely. The compiler writes this
document to the render's own scratch directory and passes libass a filename the
*server* chose; the text is parsed as text by a subtitle renderer whose job is
text. The only thing that has to be escaped afterwards is the path, and the path
is not user input.

libass also gives the preset vocabulary for free -- font, size, weight, colours,
outline, shadow, alignment, margins are all fields of an ASS style -- and it
shapes Vietnamese correctly through fribidi and harfbuzz, which a hand-rolled
``drawtext`` per cue would not.
"""

from __future__ import annotations

from visionforge.domain.subtitles import StylePreset, SubtitleTrack, clean_text

#: The resolution an ASS document declares it was authored for.
#:
#: libass scales everything -- sizes, outlines, margins -- from this to the real
#: frame, so a preset written once in 1080-line units looks the same on a 720p
#: preview and a 4K master. Without it a 48pt subtitle is half the frame height
#: on a phone render and a caption on a television.
PLAY_RES_X = 1920
PLAY_RES_Y = 1080


def _colour(rgb: tuple[int, int, int], alpha: int = 0) -> str:
    """ASS colour literal: ``&HAABBGGRR``, alpha inverted and byte-reversed.

    Two quirks in one format, kept in one function: the channel order is
    backwards from every other notation, and ``00`` means opaque.
    """
    red, green, blue = rgb
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def _timestamp(ms: int) -> str:
    """``H:MM:SS.cc`` -- ASS resolves to centiseconds and truncates below that.

    Rounding to the nearest centisecond rather than truncating: a cue asked to
    start at 1.499 s belongs nearer 1.50 than 1.49, and the systematic
    half-frame earliness of truncation is visible on fast cutting.
    """
    total = max(0, int(round(ms / 10.0)))
    hours, rest = divmod(total, 360_000)
    minutes, rest = divmod(rest, 6_000)
    seconds, centis = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def _style_line(preset: StylePreset, alignment: int) -> str:
    """One ASS ``Style:`` row, built entirely from server-side values."""
    return ",".join(
        [
            "Style: Default",
            preset.font,
            str(preset.size),
            _colour(preset.primary),
            _colour(preset.primary),  # secondary: karaoke only, unused
            _colour(preset.outline_colour),
            _colour((0, 0, 0), alpha=0x60),  # back/shadow colour
            "-1" if preset.bold else "0",
            "0",  # italic
            "0",  # underline
            "0",  # strikeout
            "100",  # scale x
            "100",  # scale y
            "0",  # spacing
            "0",  # angle
            # 1 = outline + drop shadow. 3 would draw an opaque box behind the
            # text, which none of the presets ask for.
            "1",
            f"{preset.outline:g}",
            f"{preset.shadow:g}",
            str(alignment),
            str(preset.margin_h),
            str(preset.margin_h),
            str(preset.margin_v),
            # Encoding 1 = default. Not a codepage: libass reads UTF-8 and this
            # field only selects a legacy Windows charset for fonts that need
            # one, which none of ours do.
            "1",
        ]
    )


def build_ass(track: SubtitleTrack) -> str:
    """The complete ASS document for a subtitle track.

    Every cue's text is passed through ``clean_text`` again here. It was already
    cleaned before storage, and doing it twice is deliberate: this function is
    the last point before the bytes become a file, and a document that is only
    safe because something upstream behaved is a document that is unsafe the
    first time something upstream changes.
    """
    preset = track.preset
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        # Subtitles must not stretch with the frame; libass letterboxes its own
        # coordinate space instead.
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        _style_line(preset, track.alignment),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in track.ordered:
        text = clean_text(cue.text)
        if not text:
            continue
        lines.append(
            "Dialogue: 0,"
            f"{_timestamp(cue.start_ms)},{_timestamp(cue.end_ms)},"
            "Default,,0,0,0,,"
            f"{text}"
        )

    return "\n".join(lines) + "\n"


def escape_filter_path(path: str) -> str:
    """Escape a path for use inside a filtergraph option.

    Only ever applied to a path this process generated, never to user input --
    but a Windows path contains both ``\\`` and ``:``, which are the escape
    character and the option separator, so even a server-chosen path has to be
    quoted correctly or the graph fails to parse.

    Backslashes become forward slashes first -- FFmpeg accepts them on Windows
    and it removes one whole level of escaping -- and the drive colon is then
    escaped *once*, because inside a quoted option value a colon still separates
    options. One backslash, not two: two is what the first attempt emitted, and
    FFmpeg answered ``No option name near '/Users/...'``.

    Quotes are not handled, because the paths this is given cannot contain one:
    they are a scratch directory this process created plus a fixed filename. A
    path that could carry a quote would need the option built differently, not
    escaped harder.
    """
    return path.replace("\\", "/").replace(":", "\\:")


__all__ = ["PLAY_RES_X", "PLAY_RES_Y", "build_ass", "escape_filter_path"]
