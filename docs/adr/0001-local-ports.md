# ADR-0001: Non-default host ports for local infrastructure

**Status:** Accepted · 2026-09-11 (Phase 1)

## Context

The development machine already runs services that occupy the default ports:

| Port | Held by                                  |
|------|------------------------------------------|
| 5432 | `rok_postgres` container (another project) |
| 5433 | A native PostgreSQL install              |
| 6379 | A native `redis-server` process          |

Phase 1 must not disturb any of these. VisionForge shares this machine with
several unrelated projects, and "stop the other container" is not an acceptable
prerequisite for starting work.

## Decision

VisionForge publishes its containers on non-default host ports:

| Service  | Container port | Host port |
|----------|----------------|-----------|
| Postgres | 5432           | **5442**  |
| Redis    | 6379           | **6389**  |
| MinIO    | 9000 / 9001    | 9000 / 9001 (free) |

Every port is a `.env` variable with the non-default value as its default, so a
developer on a clean machine can move them back without editing compose.

## Consequences

- Nothing on this machine has to be stopped to work on VisionForge.
- Connection strings in `.env.example` are correct out of the box.
- The published port is invisible inside the compose network: containers still
  talk to each other on 5432/6379. Only host-side clients see 5442/6389.
- In AWS the ports are the defaults again, supplied by the environment.
