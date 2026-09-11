# ADR-0004: Windows event-loop policy and a synchronous Alembic engine

**Status:** Accepted · 2026-09-11 (Phase 1)

## Context

Two problems surfaced on Windows during Phase 1:

1. psycopg's async driver cannot run on `ProactorEventLoop`, which is the Python
   default on Windows. Any async database connection fails with
   `Psycopg cannot use the 'ProactorEventLoop' to run in async mode`.
2. The policy must be installed **before** the loop is created. Running
   `uvicorn visionforge.api.main:app` does not work, because uvicorn creates the
   loop before importing the app module.

## Decision

1. `visionforge.core.runtime.configure_event_loop_policy()` installs
   `WindowsSelectorEventLoopPolicy` on `win32` and is a no-op elsewhere.
2. `python -m visionforge` is the development entrypoint. It calls the fix, then
   hands off to `uvicorn.run()`. `tests/conftest.py` calls it too.
3. Alembic uses a **synchronous** engine. Migrations are a one-shot CLI operation
   with no concurrency to exploit, so the async machinery bought nothing and cost
   a platform dependency. This also halved `alembic/env.py`.

No driver rewrite is needed for (3): `postgresql+psycopg` is psycopg3, whose
single dialect backs both the sync engine used by Alembic and the async engine
used by the application. That dual-mode support is the reason psycopg3 was chosen
over psycopg2.

## Consequences

- The workaround is one function in one file, named for what it is, and inert on
  Linux (CI and production).
- Nobody has to rediscover this: the failure mode is documented here and the
  entrypoint that avoids it is the documented way to run the API.
- On Linux, `uvicorn visionforge.api.main:app` still works directly.
