"""FFmpeg / ffprobe adapter.

Never imported by ``visionforge.api`` -- enforced by the
``api-never-touches-media`` import-linter contract.
"""

from visionforge.infra.ffmpeg.probe import probe_media
from visionforge.infra.ffmpeg.runner import FFmpegNotAvailableError, assert_ffmpeg_available, run
from visionforge.infra.ffmpeg.transcode import make_proxy, make_thumbnail

__all__ = [
    "FFmpegNotAvailableError",
    "assert_ffmpeg_available",
    "make_proxy",
    "make_thumbnail",
    "probe_media",
    "run",
]
