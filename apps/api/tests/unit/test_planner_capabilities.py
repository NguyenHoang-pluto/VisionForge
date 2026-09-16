"""What the server declares it can do.

The editor renders its controls entirely from this response: which modes to
offer, which styles, which frame rates, and what a trim handle may be dragged
to. A field that is wrong here is a control that lies, so the shape is asserted
rather than assumed -- and `segment_bounds` in particular is asserted to equal
the constants the *validator* uses, because the whole point of declaring them is
that the browser's copy cannot drift.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from visionforge.api.dependencies import get_llm_provider
from visionforge.api.main import create_app
from visionforge.domain.editplan import (
    MAX_OUTPUT_MS,
    MAX_SEGMENT_MS,
    MAX_SEGMENTS,
    MAX_TRANSITION_MS,
    MAX_TRANSITION_SHARE,
    MIN_OUTPUT_MS,
    MIN_SEGMENT_MS,
    MIN_TRANSITION_MS,
    TransitionKind,
)
from visionforge.domain.effects import EFFECT_BOUNDS, MAX_EFFECTS_PER_SEGMENT, EffectKind
from visionforge.domain.style import FPS_PRESETS
from visionforge.domain.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUE_MS,
    MAX_CUES,
    MIN_CUE_MS,
    SubtitlePosition,
    SubtitleStyle,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """No provider configured: the state a developer without a key is in."""
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def capabilities(client: TestClient) -> dict:
    response = client.get("/api/planner/capabilities")
    assert response.status_code == 200, response.text
    return dict(response.json())


class TestPlannerCapabilities:
    def test_says_plainly_that_there_is_no_model(self, capabilities: dict) -> None:
        assert capabilities["ai_available"] is False
        assert capabilities["model"] is None

    def test_offers_every_mode_even_without_a_provider(self, capabilities: dict) -> None:
        """The AI mode is still *listed*; the UI disables it from `ai_available`.

        Omitting it would make "AI is off here" indistinguishable from "this
        build has no AI", which are different things to tell a user.
        """
        assert set(capabilities["modes"]) == {"automatic", "rules", "ai"}

    def test_declares_the_offered_presets(self, capabilities: dict) -> None:
        assert capabilities["fps_presets"] == list(FPS_PRESETS)
        assert set(capabilities["quality_presets"]) == {"draft", "balanced", "high"}
        assert {item["value"] for item in capabilities["aspect_ratios"]} == {
            "16:9",
            "9:16",
            "1:1",
        }

    def test_every_aspect_ratio_carries_even_dimensions(self, capabilities: dict) -> None:
        """H.264 4:2:0 cannot encode an odd dimension."""
        for item in capabilities["aspect_ratios"]:
            assert item["width"] % 2 == 0
            assert item["height"] % 2 == 0

    def test_every_style_carries_what_a_control_needs(self, capabilities: dict) -> None:
        assert len(capabilities["styles"]) >= 8
        for style in capabilities["styles"]:
            assert style["label"]
            assert style["description"]
            assert style["default_duration_ms"] > 0
            assert style["min_clip_ms"] < style["max_clip_ms"]

    def test_segment_bounds_match_the_validator(self, capabilities: dict) -> None:
        """The reason this field exists.

        The timeline clamps a drag against these numbers in the browser. If they
        ever stop matching the constants `validate_plan` enforces, the editor
        starts permitting edits the server will reject -- which is the failure
        this assertion exists to catch at build time rather than at render time.
        """
        assert capabilities["segment_bounds"] == {
            "min_clip_ms": MIN_SEGMENT_MS,
            "max_clip_ms": MAX_SEGMENT_MS,
            "max_clips": MAX_SEGMENTS,
            "min_total_ms": MIN_OUTPUT_MS,
            "max_total_ms": MAX_OUTPUT_MS,
        }

    def test_never_contains_a_credential(self, client: TestClient) -> None:
        """A capabilities endpoint is a tempting place to leak one."""
        body = client.get("/api/planner/capabilities").text.lower()

        for forbidden in ("api_key", "apikey", "secret", "token", "authorization", "sk-"):
            assert forbidden not in body


class TestPhase9Vocabulary:
    """The transitions, effects and subtitle presets the editor draws from.

    Same argument as ``segment_bounds``: the inspector greys out a transition
    the clips are too short for and clamps a slider while the pointer is
    moving, so it needs these numbers client-side. Asserting them against the
    domain's own tables is what stops the browser's copy from drifting.
    """

    def test_every_transition_the_domain_has_is_declared(self, capabilities: dict) -> None:
        declared = {item["value"] for item in capabilities["transitions"]}

        assert declared == {kind.value for kind in TransitionKind}

    def test_each_transition_says_whether_it_costs_time(self, capabilities: dict) -> None:
        """The single question the editor's arithmetic asks."""
        by_value = {item["value"]: item for item in capabilities["transitions"]}

        assert by_value["crossfade"]["consumes_time"] is True
        assert by_value["crossfade"]["needs_previous"] is True
        assert by_value["cut"]["consumes_time"] is False
        assert by_value["fade_in"]["consumes_time"] is False

    def test_transition_bounds_match_the_validator(self, capabilities: dict) -> None:
        for item in capabilities["transitions"]:
            if item["value"] == "cut":
                assert item["min_ms"] == 0 and item["max_ms"] == 0
            else:
                assert item["min_ms"] == MIN_TRANSITION_MS
                assert item["max_ms"] == MAX_TRANSITION_MS
            assert item["max_share"] == MAX_TRANSITION_SHARE

    def test_every_effect_kind_is_declared_with_its_range(self, capabilities: dict) -> None:
        declared = {item["kind"]: item for item in capabilities["effects"]}

        assert set(declared) == {kind.value for kind in EffectKind}
        for kind in EffectKind:
            low, high, neutral = EFFECT_BOUNDS[kind]
            item = declared[kind.value]
            assert item["minimum"] == low
            assert item["maximum"] == high
            assert item["neutral"] == neutral
            assert item["whole_segment_only"] is kind.spans_whole_segment

    def test_the_neutral_value_of_every_effect_is_inside_its_range(
        self, capabilities: dict
    ) -> None:
        """A slider whose off position is outside its track cannot be turned off."""
        for item in capabilities["effects"]:
            assert item["minimum"] <= item["neutral"] <= item["maximum"]

    def test_every_subtitle_preset_is_described(self, capabilities: dict) -> None:
        declared = {item["id"] for item in capabilities["subtitle_styles"]}

        assert declared == {style.value for style in SubtitleStyle}
        for item in capabilities["subtitle_styles"]:
            assert item["label"]
            assert item["font"]
            assert item["size"] > 0

    def test_a_preset_declares_a_font_family_and_never_a_path(self, capabilities: dict) -> None:
        """The client is told which family the server chose, not where it lives.

        A path here would be a path the client could send back, and the whole
        design of the preset table is that it cannot.
        """
        for item in capabilities["subtitle_styles"]:
            assert "/" not in item["font"]
            assert "\\" not in item["font"]
            assert "." not in item["font"]

    def test_every_position_is_a_name_not_a_coordinate(self, capabilities: dict) -> None:
        assert set(capabilities["subtitle_positions"]) == {
            position.value for position in SubtitlePosition
        }

    def test_subtitle_bounds_match_the_validator(self, capabilities: dict) -> None:
        assert capabilities["subtitle_bounds"] == {
            "min_cue_ms": MIN_CUE_MS,
            "max_cue_ms": MAX_CUE_MS,
            "max_chars": MAX_CUE_CHARS,
            "max_cues": MAX_CUES,
            "max_effects_per_clip": MAX_EFFECTS_PER_SEGMENT,
        }
