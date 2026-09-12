# ADR-0005: Presigned direct-to-storage upload

**Status:** Accepted · 2026-09-12 (Phase 2)

## Context

Media files reach 2 GiB. Routing them through FastAPI would mean the API process
buffers or streams gigabytes, holds a worker for minutes per upload, and needs
memory proportional to concurrent uploads -- on a host with 7.4 GB of RAM.

It would also break the Phase 0 rule that the API never touches media bytes.

## Decision

Three steps, and the bytes never touch the API:

1. `POST /api/projects/{id}/media/upload-url` creates a `media_assets` row in
   `pending_upload` and returns a presigned `PUT` URL with a 15-minute TTL.
2. The browser `PUT`s the file directly to MinIO/S3.
3. `POST .../complete` verifies the object exists and is a sane size, marks the
   asset `uploaded`, and dispatches an ingest job.

**Object keys are derived, never supplied.** The key is
`projects/{project_id}/media/{media_id}/original{ext}` -- UUIDs only. The client's
filename is stored as a display string and is never a path component. Path
traversal is therefore impossible by construction rather than filtered for.

The size limit is enforced by the storage service through the presigned policy,
not by application code: there is no application code in the byte path to check.

## Consequences

- The API stays responsive regardless of upload size or count.
- Upload bandwidth scales with the storage service, not with the API.
- **Cost:** the server cannot inspect bytes as they arrive, so validation moves
  to `complete` (existence, size) and to the worker (magic bytes, ffprobe). A
  row can exist in `pending_upload` forever if the client abandons the upload;
  an object-lifecycle rule and a sweeper will collect those.
- A media row exists before its bytes do, which is why `pending_upload` is a
  distinct state rather than an implicit one.
