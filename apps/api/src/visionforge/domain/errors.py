"""Error taxonomy.

The split that matters is ``TransientError`` versus ``PermanentError``: it is the
single input to the retry decision. A corrupt upload must fail immediately; a
storage timeout must be retried. Everything else is detail.
"""

from __future__ import annotations

from uuid import UUID


class VisionForgeError(Exception):
    """Base class for all domain errors."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


# --------------------------------------------------------------- request errors
class NotFoundError(VisionForgeError):
    code = "not_found"
    http_status = 404


class ValidationError(VisionForgeError):
    code = "validation_error"
    http_status = 422


class ConflictError(VisionForgeError):
    code = "conflict"
    http_status = 409


class AuthorizationError(VisionForgeError):
    """The caller may not act on this resource.

    Phase 2 has no authentication, but every access still resolves through
    project ownership so that Phase 11 can swap the principal source without
    touching the domain.
    """

    code = "forbidden"
    http_status = 403


# ------------------------------------------------------------ processing errors
class TransientError(VisionForgeError):
    """Worth retrying: network blips, storage timeouts, transient locks."""

    code = "transient_error"


class PermanentError(VisionForgeError):
    """Never retry: corrupt media, unsupported codec, failed validation."""

    code = "permanent_error"


class UnsupportedMediaError(PermanentError):
    code = "unsupported_media"
    http_status = 422


class MediaTooLargeError(PermanentError):
    code = "media_too_large"
    http_status = 413


class DuplicateMediaError(VisionForgeError):
    """The uploaded bytes already exist in this project.

    Not a failure: the caller gets the asset that already holds these bytes. It
    is its own type so the task layer can resolve it without string-matching a
    generic conflict.
    """

    code = "duplicate_media"
    http_status = 409

    def __init__(self, message: str, *, duplicate_of: UUID) -> None:
        super().__init__(message)
        self.duplicate_of = duplicate_of


class CancelledError(VisionForgeError):
    """Raised inside a worker when cancellation is observed at a step boundary."""

    code = "cancelled"
