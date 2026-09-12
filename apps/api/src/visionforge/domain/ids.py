"""Typed identifiers.

Distinct ``NewType`` aliases over ``UUID`` so that a ``MediaId`` cannot be passed
where a ``ProjectId`` is expected. Free at runtime, caught by mypy.
"""

from __future__ import annotations

import uuid
from typing import NewType
from uuid import UUID

UserId = NewType("UserId", UUID)
ProjectId = NewType("ProjectId", UUID)
MediaId = NewType("MediaId", UUID)
DerivativeId = NewType("DerivativeId", UUID)
JobId = NewType("JobId", UUID)


def new_uuid() -> UUID:
    return uuid.uuid4()
