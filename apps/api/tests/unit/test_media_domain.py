"""Media validation, object-key construction and metadata rules."""

from __future__ import annotations

import uuid

import pytest

from visionforge.domain.ids import MediaId, ProjectId
from visionforge.domain.media import (
    PROXY_HEIGHT,
    SUPPORTED_EXTENSIONS,
    DerivativeKind,
    MediaKind,
    MediaMetadata,
    sniff_kind,
)
from visionforge.domain.storage import derivative_key, original_key


class TestMagicByteSniffing:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            (b"\xff\xd8\xff\xe0" + b"\x00" * 12, MediaKind.IMAGE),  # JPEG
            (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, MediaKind.IMAGE),  # PNG
            (b"GIF89a" + b"\x00" * 10, MediaKind.IMAGE),
            (b"\x00\x00\x00\x20ftypisom", MediaKind.VIDEO),  # MP4
            (b"\x1aE\xdf\xa3" + b"\x00" * 12, MediaKind.VIDEO),  # Matroska
            (b"ID3\x04" + b"\x00" * 12, MediaKind.AUDIO),  # MP3
            (b"fLaC" + b"\x00" * 12, MediaKind.AUDIO),
            (b"OggS" + b"\x00" * 12, MediaKind.AUDIO),
        ],
    )
    def test_recognises_real_signatures(self, header: bytes, expected: MediaKind) -> None:
        assert sniff_kind(header) is expected

    def test_riff_container_is_disambiguated_by_fourcc(self) -> None:
        """RIFF alone is ambiguous: WEBP is an image, WAVE is audio."""
        assert sniff_kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is MediaKind.IMAGE
        assert sniff_kind(b"RIFF\x00\x00\x00\x00WAVEfmt ") is MediaKind.AUDIO
        assert sniff_kind(b"RIFF\x00\x00\x00\x00AVI LIST") is None

    @pytest.mark.parametrize(
        "header",
        [
            b"",
            b"not media at all",
            b"<?php echo 1; ?>",
            b"MZ\x90\x00" + b"\x00" * 12,  # a Windows executable
            b"PK\x03\x04" + b"\x00" * 12,  # a zip archive
        ],
    )
    def test_rejects_non_media(self, header: bytes) -> None:
        assert sniff_kind(header) is None

    def test_a_renamed_executable_is_still_rejected(self) -> None:
        """The extension says .mp4; the bytes say otherwise. The bytes win."""
        assert sniff_kind(b"MZ\x90\x00" + b"\x00" * 12) is None


class TestObjectKeys:
    def test_keys_are_derived_from_ids_only(self) -> None:
        project_id = ProjectId(uuid.uuid4())
        media_id = MediaId(uuid.uuid4())

        key = original_key(project_id, media_id, ".mp4")
        assert key == f"projects/{project_id}/media/{media_id}/original.mp4"

    def test_derivative_keys_are_namespaced_under_the_media(self) -> None:
        project_id = ProjectId(uuid.uuid4())
        media_id = MediaId(uuid.uuid4())

        thumb = derivative_key(project_id, media_id, DerivativeKind.THUMBNAIL, "default", ".jpg")
        proxy = derivative_key(project_id, media_id, DerivativeKind.PROXY, "720p", ".mp4")

        assert thumb.endswith("/thumbnail/default.jpg")
        assert proxy.endswith("/proxy/720p.mp4")
        assert thumb.startswith(f"projects/{project_id}/media/{media_id}/")

    def test_no_client_string_reaches_the_key(self) -> None:
        """Traversal is impossible by construction: keys contain only UUIDs.

        There is no parameter through which a filename could become a path
        component, so there is nothing to sanitise.
        """
        project_id = ProjectId(uuid.uuid4())
        media_id = MediaId(uuid.uuid4())
        key = original_key(project_id, media_id, ".jpg")

        assert ".." not in key
        assert not key.startswith("/")
        assert key.count("/") == 4


class TestProxyDecision:
    def test_tall_video_needs_a_proxy(self) -> None:
        assert MediaMetadata(kind=MediaKind.VIDEO, height=1080).needs_proxy

    def test_video_at_or_below_proxy_height_does_not(self) -> None:
        assert not MediaMetadata(kind=MediaKind.VIDEO, height=PROXY_HEIGHT).needs_proxy
        assert not MediaMetadata(kind=MediaKind.VIDEO, height=480).needs_proxy

    def test_images_and_audio_never_get_a_proxy(self) -> None:
        assert not MediaMetadata(kind=MediaKind.IMAGE, height=4000).needs_proxy
        assert not MediaMetadata(kind=MediaKind.AUDIO).needs_proxy


class TestSupportedExtensions:
    def test_covers_all_three_kinds(self) -> None:
        assert set(SUPPORTED_EXTENSIONS.values()) == set(MediaKind)

    def test_every_extension_is_lowercase_and_dotted(self) -> None:
        for extension in SUPPORTED_EXTENSIONS:
            assert extension.startswith(".")
            assert extension == extension.lower()
