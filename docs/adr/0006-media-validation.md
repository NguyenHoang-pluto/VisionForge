# ADR-0006: Three-layer media validation

**Status:** Accepted · 2026-09-12 (Phase 2)

## Context

Every uploaded file is untrusted input to a C decoder. Neither the file
extension nor the client's `Content-Type` header carries any authority: both are
attacker-controlled strings.

## Decision

Validation happens in three layers, cheapest first:

| Layer | Where | Cost | Rejects |
|---|---|---|---|
| 1. Extension allow-list | API, at `upload-url` | free | `.exe`, `.php`, `.zip`, unknown types |
| 2. Magic bytes | Worker, `VALIDATE` | one ranged GET (16 bytes) | renamed executables, archives, junk |
| 3. ffprobe | Worker, `METADATA` | one subprocess | corrupt, truncated or undecodable media |

Layer 2 uses a **ranged** GET, so validating a 2 GB upload costs 16 bytes of
transfer, not 2 GB, before we know it is junk.

**ffprobe is the source of truth for metadata**, and it overrides the
extension-derived guess: a file named `.jpg` that ffprobe reports as video is
recorded as video.

## The gap ffprobe alone leaves

ffprobe is necessary but not sufficient. Its `image2` demuxer infers a codec from
the *filename extension*, so a renamed executable named `corrupt.jpg` comes back
as a successfully-probed `mjpeg` image with `0x0` dimensions rather than an
error. `probe_media` therefore rejects visual media without usable dimensions.
Without that check the file would pass `METADATA` and fail later inside FFmpeg
with a cryptic message. This is covered by
`test_corrupt_file_is_rejected`.

## Subprocess posture

Every FFmpeg and ffprobe invocation is an **argv array**; `shell=True` appears
nowhere. Each run has a hard timeout (60 s probe, 120 s thumbnail, 30 min proxy)
and its output is captured. A filename containing shell metacharacters resolves
to "no such file", not to execution -- asserted by
`test_commands_are_argv_arrays_not_shell_strings`.

## Consequences

- Junk is rejected before any expensive work, and with an actionable message.
- Three layers means three places to update when a format is added; the
  extension map and the magic-signature table live next to each other in
  `domain/media.py` to keep that honest.
