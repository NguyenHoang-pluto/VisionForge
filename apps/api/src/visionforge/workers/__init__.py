"""Worker entrypoints.

Phase 0 decided on a modular monolith: workers share the domain code with the
API and differ only in which queue they consume and which pool they use. They are
therefore modules here, not separate top-level projects.

Run them with, for example::

    celery -A visionforge.workers.cpu worker --pool=threads --concurrency=2 -Q cpu
"""
