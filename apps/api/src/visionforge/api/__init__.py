"""Interface layer: HTTP routing, request/response schemas, middleware.

Phase 0 rule #1: this package must never import FFmpeg, PyTorch or model code.
Enforced by the ``api-never-touches-media`` import-linter contract.
"""
