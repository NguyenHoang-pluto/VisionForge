"""Subprocess execution for FFmpeg and ffprobe.

Security posture (Phase 0 rule): every invocation is an **argv array**, never a
shell string. There is no code path where user input becomes a shell token. Each
run has a hard timeout and its output is captured, not streamed to a terminal.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from visionforge.domain.errors import PermanentError, TransientError

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Minimum supported major version. Asserted at worker start-up so that a broken
#: or missing install is a clear error at boot rather than a cryptic failure
#: inside job 400.
MIN_MAJOR_VERSION = 6


class FFmpegNotAvailableError(RuntimeError):
    """FFmpeg or ffprobe is missing, or too old."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run(args: list[str], *, timeout_s: float, binary: str = FFMPEG) -> CommandResult:
    """Run ffmpeg/ffprobe with an argv array and a hard timeout.

    Raises ``TransientError`` on timeout (the machine may simply be loaded) and
    ``PermanentError`` on a non-zero exit (the input is usually the problem).
    """
    executable = shutil.which(binary)
    if executable is None:
        raise FFmpegNotAvailableError(f"{binary} not found on PATH")

    argv = [executable, *args]
    logger.debug("running", extra={"binary": binary, "argc": len(argv)})

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"{binary} timed out after {timeout_s}s") from exc

    result = CommandResult(
        args=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise PermanentError(
            f"{binary} exited with {result.returncode}",
            hint=tail or None,
        )
    return result


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version``. Cached: it cannot change mid-process."""
    return run(["-version"], timeout_s=15, binary=FFMPEG).stdout.splitlines()[0]


def assert_ffmpeg_available() -> str:
    """Verify both binaries exist and are new enough. Returns the version banner.

    Called at worker start-up. Failing here is the point: a worker that cannot
    render should not accept jobs it will fail one by one.
    """
    for binary in (FFMPEG, FFPROBE):
        if shutil.which(binary) is None:
            raise FFmpegNotAvailableError(
                f"{binary} not found on PATH. "
                "Install FFmpeg and add its bin directory to PATH -- see README."
            )

    banner = ffmpeg_version()
    try:
        # "ffmpeg version 9.0.1-essentials_build-www.gyan.dev ..."
        raw = banner.split()[2]
        major = int(raw.split(".")[0].lstrip("n"))
    except (IndexError, ValueError):
        logger.warning("could not parse ffmpeg version", extra={"banner": banner})
        return banner

    if major < MIN_MAJOR_VERSION:
        raise FFmpegNotAvailableError(
            f"ffmpeg {major} is too old; VisionForge requires >= {MIN_MAJOR_VERSION}"
        )
    return banner
