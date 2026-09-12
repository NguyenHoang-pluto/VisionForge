"""ffprobe metadata extraction and FFmpeg derivative generation, against real files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from visionforge.domain.errors import PermanentError, UnsupportedMediaError
from visionforge.domain.media import (
    MAGIC_PROBE_BYTES,
    PROXY_HEIGHT,
    MediaKind,
    sniff_kind,
)
from visionforge.infra.ffmpeg import (
    assert_ffmpeg_available,
    make_proxy,
    make_thumbnail,
    probe_media,
)
from visionforge.infra.ffmpeg.runner import MIN_MAJOR_VERSION

pytestmark = pytest.mark.integration


def test_ffmpeg_is_available_and_new_enough() -> None:
    banner = assert_ffmpeg_available()
    assert "ffmpeg version" in banner
    major = int(banner.split()[2].split(".")[0].lstrip("n"))
    assert major >= MIN_MAJOR_VERSION


class TestProbe:
    def test_image_metadata(self, media_fixtures: dict[str, Path]) -> None:
        metadata = probe_media(str(media_fixtures["image.jpg"]))

        assert metadata.kind is MediaKind.IMAGE
        assert (metadata.width, metadata.height) == (64, 48)
        assert metadata.codec is not None
        assert not metadata.needs_proxy

    def test_video_metadata(self, media_fixtures: dict[str, Path]) -> None:
        metadata = probe_media(str(media_fixtures["video_1080.mp4"]))

        assert metadata.kind is MediaKind.VIDEO
        assert (metadata.width, metadata.height) == (1920, 1080)
        assert metadata.duration_ms is not None and 500 < metadata.duration_ms < 2000
        assert metadata.fps == pytest.approx(10.0, abs=0.5)
        assert metadata.codec == "h264"
        assert metadata.pix_fmt == "yuv420p"
        assert metadata.container_format is not None
        assert metadata.needs_proxy

    def test_audio_metadata(self, media_fixtures: dict[str, Path]) -> None:
        metadata = probe_media(str(media_fixtures["audio.mp3"]))

        assert metadata.kind is MediaKind.AUDIO
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.duration_ms is not None
        assert metadata.width is None
        assert not metadata.needs_proxy

    def test_stream_layout_is_captured(self, media_fixtures: dict[str, Path]) -> None:
        metadata = probe_media(str(media_fixtures["video_720.mp4"]))
        assert any(s.kind == "video" for s in metadata.streams)

    def test_720p_video_does_not_need_a_proxy(self, media_fixtures: dict[str, Path]) -> None:
        assert not probe_media(str(media_fixtures["video_720.mp4"])).needs_proxy

    def test_corrupt_file_is_rejected(self, media_fixtures: dict[str, Path]) -> None:
        """A renamed executable must not survive probing.

        Two layers catch this. Magic-byte sniffing rejects it before ffprobe is
        ever spawned; and ffprobe alone is *not* sufficient, because its image2
        demuxer infers "mjpeg" from the .jpg extension and returns a 0x0 stream
        instead of failing. The dimension check in ``probe_media`` is what turns
        that into a clear rejection.
        """
        path = media_fixtures["corrupt.jpg"]

        assert sniff_kind(path.read_bytes()[:MAGIC_PROBE_BYTES]) is None

        with pytest.raises((UnsupportedMediaError, PermanentError)):
            probe_media(str(path))

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PermanentError):
            probe_media(str(tmp_path / "does-not-exist.mp4"))


class TestDerivatives:
    def test_image_thumbnail(self, media_fixtures: dict[str, Path], tmp_path: Path) -> None:
        source = media_fixtures["image.jpg"]
        destination = tmp_path / "thumb.jpg"

        make_thumbnail(str(source), str(destination), probe_media(str(source)), width=32)

        assert destination.exists() and destination.stat().st_size > 0
        assert probe_media(str(destination)).width == 32

    def test_video_thumbnail_is_a_single_frame(
        self, media_fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        source = media_fixtures["video_1080.mp4"]
        destination = tmp_path / "thumb.jpg"

        make_thumbnail(str(source), str(destination), probe_media(str(source)), width=48)

        metadata = probe_media(str(destination))
        assert metadata.kind is MediaKind.IMAGE
        assert metadata.width == 48

    def test_proxy_is_scaled_to_720p(self, media_fixtures: dict[str, Path], tmp_path: Path) -> None:
        source = media_fixtures["video_1080.mp4"]
        destination = tmp_path / "proxy.mp4"

        make_proxy(str(source), str(destination), height=PROXY_HEIGHT)

        metadata = probe_media(str(destination))
        assert metadata.kind is MediaKind.VIDEO
        assert metadata.height == PROXY_HEIGHT
        assert metadata.width == 1280  # aspect ratio preserved
        assert metadata.codec == "h264"

    def test_original_is_never_modified(
        self, media_fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        source = media_fixtures["video_1080.mp4"]
        before = source.read_bytes()

        make_proxy(str(source), str(tmp_path / "proxy.mp4"), height=PROXY_HEIGHT)
        make_thumbnail(str(source), str(tmp_path / "t.jpg"), probe_media(str(source)), width=32)

        assert source.read_bytes() == before


class TestSecurity:
    def test_commands_are_argv_arrays_not_shell_strings(self, tmp_path: Path) -> None:
        """A filename containing shell metacharacters must not be interpreted.

        The runner passes argv arrays and never ``shell=True``, so this resolves
        to "no such file" rather than executing anything.
        """
        hostile = tmp_path / "a; rm -rf x.mp4"
        hostile.write_bytes(b"\x00" * 64)

        with pytest.raises(PermanentError):
            probe_media(str(hostile))

        assert hostile.exists()

    def test_timeout_is_enforced(self) -> None:
        """A hung subprocess must be killed, not waited on forever."""
        from visionforge.domain.errors import TransientError
        from visionforge.infra.ffmpeg.runner import run

        with pytest.raises((TransientError, PermanentError, subprocess.SubprocessError)):
            # Reading from a never-ending source with a sub-second budget.
            run(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=1920x1080:rate=60",
                    "-t",
                    "600",
                    "-f",
                    "null",
                    "-",
                ],
                timeout_s=0.5,
            )
