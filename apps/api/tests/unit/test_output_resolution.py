"""Output size and frame rate: the named sizes, their pixels, and what they cost.

The client names a resolution and never a width or a height. These tests pin
the pixels each name becomes, that a shape change keeps the size, and that a
large frame is encoded with a bounded thread count and a longer time limit.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest
from pydantic import ValidationError

from visionforge.api.schemas.edit import ManualPlanCreateRequest, PlanCreateRequest
from visionforge.domain.editplan import (
    MAX_DIMENSION,
    AspectRatio,
    EditPlan,
    Encoder,
    OutputSpec,
    QualityPreset,
    Resolution,
    Segment,
)
from visionforge.domain.effects import Effect, EffectKind
from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.style import FPS_PRESETS, dimensions_for
from visionforge.infra.ffmpeg.compiler import (
    LARGE_FRAME_THREADS,
    build_filter_graph,
    compile_render_argv,
)
from visionforge.workers.video_render import (
    MAX_RENDER_TIMEOUT_S,
    RENDER_TIMEOUT_S,
    render_timeout_s,
    uses_proxy,
)

from .test_planner_and_compiler import TestRenderSpec


@pytest.mark.parametrize(
    ("aspect", "resolution", "expected"),
    [
        (AspectRatio.LANDSCAPE_16_9, Resolution.P720, (1280, 720)),
        (AspectRatio.LANDSCAPE_16_9, Resolution.P1080, (1920, 1080)),
        (AspectRatio.LANDSCAPE_16_9, Resolution.P1440, (2560, 1440)),
        (AspectRatio.LANDSCAPE_16_9, Resolution.P2160, (3840, 2160)),
        (AspectRatio.PORTRAIT_9_16, Resolution.P2160, (2160, 3840)),
        (AspectRatio.SQUARE_1_1, Resolution.P1080, (1080, 1080)),
    ],
)
def test_a_named_size_becomes_the_expected_pixels(
    aspect: AspectRatio, resolution: Resolution, expected: tuple[int, int]
) -> None:
    assert dimensions_for(aspect, resolution) == expected


def test_every_offered_size_passes_the_plan_validator_bounds() -> None:
    for aspect in AspectRatio:
        for resolution in Resolution:
            width, height = dimensions_for(aspect, resolution)
            assert width <= MAX_DIMENSION and height <= MAX_DIMENSION
            assert width % 2 == 0 and height % 2 == 0


def test_the_default_size_is_unchanged() -> None:
    """A request that names no size renders at 720p, as it always has."""
    assert PlanCreateRequest().resolution is Resolution.P720


def test_120_fps_is_offered_and_accepted() -> None:
    assert max(FPS_PRESETS) == 120
    assert PlanCreateRequest(fps=120).fps == 120
    assert (
        ManualPlanCreateRequest.model_validate(
            {
                "segments": [
                    {
                        "media_id": "00000000-0000-0000-0000-000000000001",
                        "source_in_ms": 0,
                        "source_out_ms": 2000,
                    }
                ],
                "fps": 120,
                "resolution": "2160p",
            }
        ).resolution
        is Resolution.P2160
    )


def test_a_frame_rate_outside_the_presets_is_refused() -> None:
    with pytest.raises(ValidationError):
        PlanCreateRequest(fps=90)


def test_16k_is_not_a_resolution() -> None:
    """Past every NVENC encoder's 8192 limit, and past this machine's memory on the CPU."""
    with pytest.raises(ValidationError):
        PlanCreateRequest.model_validate({"resolution": "8640p"})


def test_8k_is_7680_by_4320() -> None:
    assert dimensions_for(AspectRatio.LANDSCAPE_16_9, Resolution.P4320) == (7680, 4320)


def test_8k_on_the_cpu_is_refused_by_the_validator() -> None:
    from visionforge.domain.editplan import MediaFact, validate_plan
    from visionforge.domain.media import MediaKind, MediaStatus

    media = MediaId(uuid.uuid4())
    project = ProjectId(uuid.uuid4())
    facts = {
        media: MediaFact.from_media(
            media_id=media,
            project_id=project,
            kind=MediaKind.VIDEO,
            status=MediaStatus.READY,
            duration_ms=10_000,
        )
    }

    def plan(encoder: Encoder) -> EditPlan:
        return EditPlan(
            project_id=project,
            segments=(Segment(media_id=media, order=0, source_in_ms=0, source_out_ms=2000),),
            output=OutputSpec(width=7680, height=4320, encoder=encoder),
        )

    assert "cpu_too_large" in {v.code for v in validate_plan(plan(Encoder.CPU), facts)}
    assert validate_plan(plan(Encoder.GPU), facts) == []


def test_the_gpu_uses_nvenc_h264_up_to_4k() -> None:
    spec = replace(TestRenderSpec()._spec(2), encoder=Encoder.GPU, crf=19, preset="p6")
    argv = compile_render_argv(spec)
    assert argv[argv.index("-c:v") + 1] == "h264_nvenc"
    assert argv[argv.index("-cq") + 1] == "19"
    assert argv[argv.index("-preset") + 1] == "p6"
    assert "-crf" not in argv and "-threads" not in argv


def test_the_gpu_switches_to_hevc_above_4096() -> None:
    from visionforge.domain.timeline import VideoCodec

    spec = replace(
        TestRenderSpec()._spec(2),
        encoder=Encoder.GPU,
        video_codec=VideoCodec.HEVC,
        width=7680,
        height=4320,
    )
    argv = compile_render_argv(spec)
    assert argv[argv.index("-c:v") + 1] == "hevc_nvenc"
    assert argv[argv.index("-tag:v") + 1] == "hvc1"


def test_a_gpu_plan_at_8k_compiles_to_hevc() -> None:
    from visionforge.domain.timeline import VideoCodec

    spec = TestRenderSpec()._spec(2, width=7680, height=4320, encoder=Encoder.GPU)
    assert spec.video_codec is VideoCodec.HEVC
    assert spec.preset.startswith("p")


def test_a_large_frame_caps_encoder_threads() -> None:
    spec = TestRenderSpec()._spec(2)
    assert "-threads" not in compile_render_argv(spec)

    large = replace(spec, width=3840, height=2160)
    argv = compile_render_argv(large)
    assert argv[argv.index("-threads") + 1] == str(LARGE_FRAME_THREADS)


def test_the_render_time_limit_grows_with_the_pixel_rate() -> None:
    assert render_timeout_s(1280, 720, 30) == RENDER_TIMEOUT_S
    assert render_timeout_s(640, 360, 24) == RENDER_TIMEOUT_S
    assert render_timeout_s(1920, 1080, 30) == pytest.approx(RENDER_TIMEOUT_S * 2.25)
    assert render_timeout_s(3840, 2160, 120) == MAX_RENDER_TIMEOUT_S


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1280, 720, True),
        (720, 1280, True),
        (720, 720, True),
        (1920, 1080, False),
        (2160, 3840, False),
    ],
)
def test_the_proxy_is_used_only_when_the_output_is_no_larger(
    width: int, height: int, expected: bool
) -> None:
    """A 720p proxy under a 1080p or 4K output would be an upscale."""
    plan = EditPlan(
        project_id=ProjectId(uuid.uuid4()),
        planner="test",
        planner_version="1",
        output=OutputSpec(width=width, height=height),
        segments=(
            Segment(media_id=MediaId(uuid.uuid4()), order=0, source_in_ms=0, source_out_ms=2000),
        ),
    )
    assert uses_proxy(plan) is expected


def test_the_fast_levels_render_exactly_as_before() -> None:
    """Bicubic, no supersampling, 128 kbps: a balanced plan's bytes are unchanged."""
    spec = TestRenderSpec()._spec(2)
    assert spec.scale_flags is None and not spec.supersample
    assert spec.audio_bitrate_kbps == 128
    assert (
        "flags=lanczos"
        not in compile_render_argv(spec)[compile_render_argv(spec).index("-filter_complex") + 1]
    )


def test_max_quality_encodes_slowly_and_resamples_with_lanczos() -> None:
    spec = TestRenderSpec()._spec(2, quality=QualityPreset.MAX)
    assert (spec.crf, spec.preset) == (16, "slow")
    assert spec.audio_bitrate_kbps == 256
    graph = compile_render_argv(spec)[compile_render_argv(spec).index("-filter_complex") + 1]
    assert "flags=lanczos" in graph


def test_a_zoomed_segment_is_supersampled_before_its_crop() -> None:
    """A 12% zoom at 1280x720 is decoded at 1434x806, so the crop is real pixels."""
    spec = TestRenderSpec()._spec(1, quality=QualityPreset.HIGH)
    zoomed = replace(
        spec,
        segments=(replace(spec.segments[0], effects=(Effect(EffectKind.ZOOM_IN, 0.12),)),),
    )
    graph = build_filter_graph(zoomed)
    assert "scale=1434:806:force_original_aspect_ratio=increase:flags=lanczos" in graph
    assert "scale=1280:720:flags=lanczos" in graph


def test_a_slower_preset_gets_a_longer_time_limit() -> None:
    assert render_timeout_s(1920, 1080, 30, "slow") > render_timeout_s(1920, 1080, 30)


def test_the_api_refuses_an_encoder_the_server_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from visionforge.api.routers import edit
    from visionforge.core.config import get_settings
    from visionforge.domain.errors import ValidationError as DomainValidationError

    monkeypatch.setenv("RENDER_GPU", "false")
    get_settings.cache_clear()
    try:
        with pytest.raises(DomainValidationError):
            edit._check_encoder(Encoder.GPU, Resolution.P1080)
        with pytest.raises(DomainValidationError):
            edit._check_encoder(Encoder.CPU, Resolution.P4320)
        edit._check_encoder(Encoder.CPU, Resolution.P2160)

        monkeypatch.setenv("RENDER_GPU", "true")
        get_settings.cache_clear()
        edit._check_encoder(Encoder.GPU, Resolution.P4320)
    finally:
        get_settings.cache_clear()
