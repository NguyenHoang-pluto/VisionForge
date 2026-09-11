"""Domain layer: entities, value objects and ports.

Pure Python. Imports nothing from ``infra`` or ``api`` and no third-party I/O
library -- enforced by the ``domain-is-pure`` import-linter contract. Phase 1
contains only what the readiness check needs; entities arrive with the media and
job modules in Phase 2.
"""
