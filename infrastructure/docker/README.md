# Production images

Empty by design in Phase 1.

Locally, only stateful services run in containers (see
[ADR-0002](../../docs/adr/0002-hybrid-local-topology.md)); the API and workers run
natively. Production images arrive in Phase 2, when there is an API worth
shipping:

- `Dockerfile.api`     — FastAPI, slim base, no media or ML dependencies
- `Dockerfile.worker`  — CPU and render workers, FFmpeg included
- `Dockerfile.gpu`     — GPU worker, CUDA base, weights mounted not baked

Writing them now would mean maintaining three Dockerfiles for an application that
has no endpoints yet.
