"""Editing styles, and the deterministic profile behind each one.

A style is a *name for a set of numbers*. "Cinematic" means long takes, an
ordering that respects the shot sequence, and a bias toward clean, well-exposed
frames; "fast montage" means short takes and a bias toward sharpness over
composure. Writing those numbers down here, rather than leaving them implicit in
a prompt, has three consequences:

- the rules engine can honour a style with no model involved, so choosing
  "Cinematic" with the AI disabled still produces a cinematic edit rather than a
  generic one -- which is what makes the fallback a fallback and not a downgrade;
- the model is given the profile as *bounds to work within*, so its pacing
  decisions are checkable against a number rather than against taste;
- a style change is a diff in this file, reviewable, instead of a reworded
  prompt whose effect nobody can predict.

No style here claims to be a template, a preset from another product, or an
imitation of anyone's edit. They are pacing and ranking parameters with
recognisable names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from visionforge.domain.editplan import AspectRatio, AudioMode, QualityPreset, Resolution
from visionforge.domain.selection import DEFAULT_SELECTION_WEIGHTS, SelectionWeights
from visionforge.domain.subtitles import SubtitlePosition, SubtitleStyle


class EditStyle(StrEnum):
    """The styles a caller may ask for. A closed set.

    ``CUSTOM`` means "the user described it in their own words" -- it carries
    neutral bounds and leaves the interpretation to the request text, which only
    the LLM planner can read. The rules engine treats it as neutral.
    """

    CINEMATIC = "cinematic"
    FAST_MONTAGE = "fast_montage"
    SPORTS_HIGHLIGHT = "sports_highlight"
    GAMING = "gaming"
    ANIME = "anime"
    NATURE = "nature"
    SOCIAL = "social"
    CUSTOM = "custom"


#: CRF and x264 preset per quality level. A closed map on the server: the client
#: names a level and never supplies an encoder argument.
#:
#: ``BALANCED`` is deliberately identical to the Phase 4 hard-coded defaults, so
#: an existing plan re-rendered under Phase 5 produces the same bytes.
QUALITY_SETTINGS: dict[QualityPreset, tuple[int, str]] = {
    QualityPreset.DRAFT: (28, "ultrafast"),
    QualityPreset.BALANCED: (23, "veryfast"),
    QualityPreset.HIGH: (19, "medium"),
    # CRF 16 is visually lossless for almost all footage; ``slow`` spends the
    # time finding it. Roughly four times the encode of ``BALANCED``.
    QualityPreset.MAX: (16, "slow"),
}

#: The same levels on the GPU: an NVENC constant-quality target and a preset
#: from ``p1`` (fastest) to ``p7`` (best). The targets match the CPU CRFs so a
#: level means about the same picture on either encoder.
GPU_QUALITY_SETTINGS: dict[QualityPreset, tuple[int, str]] = {
    QualityPreset.DRAFT: (28, "p1"),
    QualityPreset.BALANCED: (23, "p4"),
    QualityPreset.HIGH: (19, "p6"),
    QualityPreset.MAX: (16, "p7"),
}

#: How frames are resampled, per quality level. ``None`` keeps FFmpeg's default
#: (bicubic), which is what every level rendered with before Phase 12 and what
#: ``DRAFT`` and ``BALANCED`` still use, so their output is unchanged. Lanczos
#: keeps more edge detail on both upscale and downscale, at a small cost.
SCALE_FLAGS: dict[QualityPreset, str | None] = {
    QualityPreset.DRAFT: None,
    QualityPreset.BALANCED: None,
    QualityPreset.HIGH: "lanczos",
    QualityPreset.MAX: "lanczos",
}

#: Audio bitrate per quality level, in kbps. AAC at 128 is fine for a preview;
#: music heard at the quality the picture is shown at wants more.
AUDIO_KBPS: dict[QualityPreset, int] = {
    QualityPreset.DRAFT: 128,
    QualityPreset.BALANCED: 128,
    QualityPreset.HIGH: 192,
    QualityPreset.MAX: 256,
}

#: Frame rates a caller may choose. Also closed, and it goes up to the plan
#: validator's own ceiling of 120. A rate above the source's is reached by
#: repeating frames, so it only adds smoothness when the footage was shot fast.
FPS_PRESETS: tuple[int, ...] = (24, 25, 30, 48, 50, 60, 120)

#: Output geometry per aspect ratio. A closed map rather than free width/height:
#: the caller picks a shape, the server picks dimensions that are even (required
#: by H.264 4:2:0) and sane for this hardware. Arbitrary geometry from a client
#: is exactly what the plan validator would then have to defend against.
#:
#: In the domain rather than in the route that first needed it, because Phase 10
#: gave it a second caller: a ``CHANGE_OUTPUT_PRESET`` operation names a shape
#: and the patcher has to resolve it to pixels. Two copies of a table whose
#: whole purpose is that clients cannot choose geometry is one copy too many.
PRESET_DIMENSIONS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.LANDSCAPE_16_9: (1280, 720),
    AspectRatio.PORTRAIT_9_16: (720, 1280),
    AspectRatio.SQUARE_1_1: (720, 720),
}


def dimensions_for(
    aspect: AspectRatio, resolution: Resolution = Resolution.P720
) -> tuple[int, int]:
    """The pixels for a shape at a size.

    ``PRESET_DIMENSIONS`` is the shape at 720 lines; any other size is that
    shape scaled until its short side has ``resolution.lines`` lines.
    """
    return scale_to_lines(PRESET_DIMENSIONS[aspect], resolution.lines)


def scale_to_lines(size: tuple[int, int], lines: int) -> tuple[int, int]:
    """Scale ``size`` so its short side is ``lines``, keeping both sides even.

    Even because H.264 4:2:0 cannot encode an odd dimension, and the plan
    validator would reject one.
    """
    width, height = size
    short = min(width, height)
    return (
        round(width * lines / short / 2) * 2,
        round(height * lines / short / 2) * 2,
    )


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """The numbers a style stands for.

    ``min_clip_ms``/``max_clip_ms`` are *pacing bounds*, narrower than the
    plan-level ``MIN_SEGMENT_MS``/``MAX_SEGMENT_MS``. The plan limits say what is
    renderable; these say what is stylistically coherent. Both are enforced --
    the style bounds first, the plan bounds last and unconditionally.
    """

    style: EditStyle
    label: str
    #: One line, shown in the UI and given to the model as the style's definition.
    description: str

    min_clip_ms: int
    max_clip_ms: int
    #: Where an unconstrained planner should sit within those bounds.
    target_clip_ms: int

    default_duration_ms: int
    default_aspect: AspectRatio
    default_audio: AudioMode
    #: ``True`` when the style reads better in upload order than by score --
    #: anything with a narrative or a chronology.
    prefer_sequence_order: bool

    #: Selection weights for this style. Must still sum to 1.0; the
    #: ``SelectionWeights`` constructor enforces that, so a typo here fails at
    #: import rather than producing a quietly skewed ranking.
    weights: SelectionWeights = DEFAULT_SELECTION_WEIGHTS

    def clamp_clip_ms(self, value: int) -> int:
        return max(self.min_clip_ms, min(self.max_clip_ms, value))


#: A profile that reproduces Phase 4 behaviour exactly. Used when no style is
#: stated, so that "no style" is a real option rather than a hidden default.
NEUTRAL_PROFILE = StyleProfile(
    style=EditStyle.CUSTOM,
    label="Neutral",
    description="No stylistic bias. Clips ranked on technical quality alone.",
    min_clip_ms=300,
    max_clip_ms=30_000,
    target_clip_ms=5_000,
    default_duration_ms=25_000,
    default_aspect=AspectRatio.LANDSCAPE_16_9,
    default_audio=AudioMode.NONE,
    prefer_sequence_order=False,
)


STYLE_PROFILES: dict[EditStyle, StyleProfile] = {
    EditStyle.CINEMATIC: StyleProfile(
        style=EditStyle.CINEMATIC,
        label="Cinematic",
        description=(
            "Long, composed takes in shot order. Favours clean exposure and "
            "contrast over raw sharpness."
        ),
        min_clip_ms=2_500,
        max_clip_ms=8_000,
        target_clip_ms=4_500,
        default_duration_ms=40_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.SOURCE,
        prefer_sequence_order=True,
        # Exposure and contrast carry a cinematic frame; a slightly soft shot
        # that is well lit reads better than a clinical one that is blown out.
        weights=SelectionWeights(
            sharpness=0.28, exposure=0.32, contrast=0.25, resolution=0.10, duration=0.05
        ),
    ),
    EditStyle.FAST_MONTAGE: StyleProfile(
        style=EditStyle.FAST_MONTAGE,
        label="Fast montage",
        description="Short cuts, strongest material first. Built for pace.",
        min_clip_ms=600,
        max_clip_ms=2_000,
        target_clip_ms=1_100,
        default_duration_ms=20_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.NONE,
        prefer_sequence_order=False,
        # At one second a clip, softness is the failure the eye catches first.
        weights=SelectionWeights(
            sharpness=0.50, exposure=0.20, contrast=0.20, resolution=0.08, duration=0.02
        ),
    ),
    EditStyle.SPORTS_HIGHLIGHT: StyleProfile(
        style=EditStyle.SPORTS_HIGHLIGHT,
        label="Sports highlight",
        description=(
            "Short action beats in the order they happened. Keeps the crowd and "
            "commentary audio."
        ),
        min_clip_ms=1_200,
        max_clip_ms=4_000,
        target_clip_ms=2_200,
        default_duration_ms=30_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.SOURCE,
        # A highlight that reorders the match is not a highlight of the match.
        prefer_sequence_order=True,
        weights=SelectionWeights(
            sharpness=0.45, exposure=0.22, contrast=0.20, resolution=0.10, duration=0.03
        ),
    ),
    EditStyle.GAMING: StyleProfile(
        style=EditStyle.GAMING,
        label="Gaming montage",
        description="Punchy cuts from clean capture. Resolution matters more than usual.",
        min_clip_ms=800,
        max_clip_ms=3_000,
        target_clip_ms=1_600,
        default_duration_ms=25_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.SOURCE,
        prefer_sequence_order=False,
        # Screen capture is already sharp; what separates good from bad capture
        # is resolution and whether the encoder crushed the contrast.
        weights=SelectionWeights(
            sharpness=0.32, exposure=0.20, contrast=0.22, resolution=0.22, duration=0.04
        ),
    ),
    EditStyle.ANIME: StyleProfile(
        style=EditStyle.ANIME,
        label="Anime / AMV",
        description=(
            "Rhythmic mid-length cuts. Tolerates flat animation cels that a "
            "sharpness-led ranking would reject."
        ),
        min_clip_ms=900,
        max_clip_ms=3_500,
        target_clip_ms=1_800,
        default_duration_ms=30_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.NONE,
        prefer_sequence_order=False,
        # Animation has large flat regions, so Laplacian variance under-reads
        # legitimately good frames. Leaning on contrast compensates.
        weights=SelectionWeights(
            sharpness=0.25, exposure=0.25, contrast=0.35, resolution=0.12, duration=0.03
        ),
    ),
    EditStyle.NATURE: StyleProfile(
        style=EditStyle.NATURE,
        label="Nature",
        description="Slow, held shots in sequence. Rewards even exposure and detail.",
        min_clip_ms=3_000,
        max_clip_ms=10_000,
        target_clip_ms=5_500,
        default_duration_ms=45_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.SOURCE,
        prefer_sequence_order=True,
        weights=SelectionWeights(
            sharpness=0.35, exposure=0.30, contrast=0.20, resolution=0.12, duration=0.03
        ),
    ),
    EditStyle.SOCIAL: StyleProfile(
        style=EditStyle.SOCIAL,
        label="Social / vertical",
        description="Vertical, short, front-loaded. The first clip has to earn the second.",
        min_clip_ms=700,
        max_clip_ms=2_500,
        target_clip_ms=1_400,
        default_duration_ms=15_000,
        default_aspect=AspectRatio.PORTRAIT_9_16,
        default_audio=AudioMode.SOURCE,
        prefer_sequence_order=False,
        weights=SelectionWeights(
            sharpness=0.45, exposure=0.25, contrast=0.20, resolution=0.08, duration=0.02
        ),
    ),
    EditStyle.CUSTOM: StyleProfile(
        style=EditStyle.CUSTOM,
        label="Custom",
        description="Described by the user in their own words.",
        min_clip_ms=800,
        max_clip_ms=8_000,
        target_clip_ms=3_000,
        default_duration_ms=25_000,
        default_aspect=AspectRatio.LANDSCAPE_16_9,
        default_audio=AudioMode.NONE,
        prefer_sequence_order=False,
    ),
}


def profile_for(style: EditStyle | None) -> StyleProfile:
    """The profile for a style, or the neutral one when no style was stated."""
    if style is None:
        return NEUTRAL_PROFILE
    return STYLE_PROFILES[style]


# ------------------------------------------------------------- style inference
#: Keyword to style. This is a lookup table, not language understanding, and it
#: is only consulted when no model is available -- the honest degradation of
#: "interpret this request" is "match some words in it".
#:
#: Ordered most-specific first: "travel vlog" should not be caught by "vlog"
#: before "travel" has been considered.
_STYLE_KEYWORDS: tuple[tuple[EditStyle, tuple[str, ...]], ...] = (
    (
        EditStyle.SPORTS_HIGHLIGHT,
        ("football", "soccer", "basketball", "sport", "match", "goal", "highlight"),
    ),
    (
        EditStyle.GAMING,
        ("gaming", "gameplay", "montage kill", "fps", "valorant", "minecraft", "stream"),
    ),
    (EditStyle.ANIME, ("anime", "amv", "manga", "weeb")),
    (EditStyle.NATURE, ("nature", "landscape", "calm", "peaceful", "forest", "ocean", "wildlife")),
    (EditStyle.SOCIAL, ("tiktok", "reel", "short", "vertical", "story", "social")),
    (EditStyle.CINEMATIC, ("cinematic", "film", "movie", "travel", "documentary", "moody")),
    (EditStyle.FAST_MONTAGE, ("fast", "energetic", "montage", "hype", "punchy", "upbeat")),
)


def infer_style(text: str | None) -> EditStyle | None:
    """Guess a style from a request by keyword match. Deterministic.

    Returns ``None`` when nothing matches, which the caller should treat as "no
    style" rather than substituting a default -- a wrong style is worse than
    none, and the neutral profile is a defensible edit.

    This exists for automatic mode when the AI is unavailable. It is not
    presented to the user as understanding; the UI reports "matched: cinematic",
    and the plan metadata records ``style_source: "keyword"``.
    """
    if not text:
        return None
    lowered = text.lower()
    for style, keywords in _STYLE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return style
    return None


# ------------------------------------------------ style profile -> preset (§7)
#: Which look suits which edit style.
#:
#: This is the whole of the reference-style influence on subtitles, and note
#: what it maps *to*: a preset id. A cinematic reference makes the editor open
#: on the cinematic preset; it does not make the reference's own typography,
#: colours or margins reach the renderer, because there is no path by which a
#: measurement of somebody else's video could become a font size here. The
#: preset table above stays the only authority on what a look means.
#:
#: A preference, too, not a decision: whatever this returns is the value the UI
#: starts on, and the user's own choice replaces it.
STYLE_SUBTITLE_PRESET: dict[EditStyle, SubtitleStyle] = {
    EditStyle.CINEMATIC: SubtitleStyle.CINEMATIC,
    EditStyle.FAST_MONTAGE: SubtitleStyle.BOLD,
    EditStyle.SPORTS_HIGHLIGHT: SubtitleStyle.BOLD,
    EditStyle.GAMING: SubtitleStyle.BOLD,
    EditStyle.ANIME: SubtitleStyle.CLEAN,
    EditStyle.NATURE: SubtitleStyle.MINIMAL,
    EditStyle.SOCIAL: SubtitleStyle.SOCIAL,
    EditStyle.CUSTOM: SubtitleStyle.CLEAN,
}


def subtitle_style_for(style: EditStyle | None) -> SubtitleStyle:
    """The preset an edit style suggests. ``CLEAN`` when nothing was chosen."""
    if style is None:
        return SubtitleStyle.CLEAN
    return STYLE_SUBTITLE_PRESET.get(style, SubtitleStyle.CLEAN)


def subtitle_position_for(aspect: str | None) -> SubtitlePosition:
    """Where the text sits, given the output shape.

    Vertical output is the one case where the default moves: a 9:16 frame is
    played in an interface that draws its own controls over the bottom of the
    picture, so the bottom margin that is correct on a 16:9 master puts the
    line underneath a share button. Every other ratio gets the bottom.
    """
    if aspect in {"9:16", "4:5"}:
        return SubtitlePosition.CENTER
    return SubtitlePosition.BOTTOM


__all__ = [
    "AUDIO_KBPS",
    "FPS_PRESETS",
    "NEUTRAL_PROFILE",
    "PRESET_DIMENSIONS",
    "QUALITY_SETTINGS",
    "SCALE_FLAGS",
    "STYLE_PROFILES",
    "STYLE_SUBTITLE_PRESET",
    "EditStyle",
    "QualityPreset",
    "Resolution",
    "StyleProfile",
    "dimensions_for",
    "infer_style",
    "profile_for",
    "scale_to_lines",
    "subtitle_position_for",
    "subtitle_style_for",
]
