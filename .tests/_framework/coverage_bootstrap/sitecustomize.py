"""Start coverage measurement in child Python processes.

This module is put on ``PYTHONPATH`` by
:func:`_framework.coverage_utils.enable_subprocess_coverage`.  Python imports
``sitecustomize`` automatically at interpreter start-up, so every child
process inherits the measurement without having to wrap each command in
``coverage run``.  Measurement only starts when ``COVERAGE_PROCESS_START`` is
set, which keeps normal (non-coverage) test runs completely unaffected.
"""

from __future__ import annotations

import os

if os.environ.get("COVERAGE_PROCESS_START"):
    try:
        import coverage

        coverage.process_startup()
    except Exception:  # pragma: no cover - never break a child process
        # A broken coverage installation must not make a test fail.  The
        # missing data is reported by the coverage-report step later on.
        pass
