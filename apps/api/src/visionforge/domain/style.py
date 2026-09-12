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

from visionforge.domain.editplan import AspectRatio, AudioMode, QualityPreset
from visionforge.domain.selection import DEFAULT_SELECTION_WEIGHTS, SelectionWeights


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
}

#: Frame rates a caller may choose. Also closed: 120 fps is inside the plan
#: validator's bounds but is not something this hardware should be asked for
#: from a dropdown.
FPS_PRESETS: tuple[int, ...] = (24, 30, 60)


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


__all__ = [
    "FPS_PRESETS",
    "NEUTRAL_PROFILE",
    "QUALITY_SETTINGS",
    "STYLE_PROFILES",
    "EditStyle",
    "QualityPreset",
    "StyleProfile",
    "infer_style",
    "profile_for",
]
