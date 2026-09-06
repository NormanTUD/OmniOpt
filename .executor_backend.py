"""Abstraction layer over :mod:`submitit`.

Goal
----
Let OmniOpt2 run on a system that does **not** have SLURM available without
the user having to think about it.

* If ``sbatch`` is on ``$PATH``, we use :class:`submitit.AutoExecutor` (which
  transparently builds SLURM scripts and submits them with ``sbatch``).
* Otherwise we fall back to :class:`submitit.LocalExecutor` which executes
  jobs in subprocesses on the same machine, parallelised by Python's
  :mod:`concurrent.futures`.

On top of that we provide:

* :func:`is_sbatch_in_path` — a small helper used both here and elsewhere.
* :class:`CPUMonitor` — a background thread that periodically samples
  ``psutil.cpu_percent`` and ``psutil.virtual_memory`` and updates a Rich
  :class:`rich.progress.Progress` bar so the user can see how busy the
  machine is while jobs run.

The functions in this module never touch :mod:`omniopt`'s globals; they are
intentionally side-effect free so they can be unit-tested in isolation.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:
    import submitit
    from submitit import AutoExecutor, DebugJob, Job, LocalExecutor, LocalJob
except ImportError:  # pragma: no cover - submitit is required for runtime
    submitit = None  # type: ignore
    AutoExecutor = None  # type: ignore
    LocalExecutor = None  # type: ignore
    Job = None  # type: ignore
    LocalJob = None  # type: ignore
    DebugJob = None  # type: ignore

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:  # pragma: no cover
    Console = None  # type: ignore
    Live = None  # type: ignore
    Progress = None  # type: ignore
    BarColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore


__all__ = [
    "Backend",
    "BackendKind",
    "CPUMonitor",
    "is_sbatch_in_path",
    "make_backend",
]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Re-export :func:`.helpers.is_sbatch_in_path` so callers can keep using
# ``executor_backend.is_sbatch_in_path`` without us duplicating the logic.
# We don't redefine the function here on purpose so that the duplicate-
# function detector in .tests/ stays happy.  If the import fails we keep a
# reference to the error and use a private fallback instead.
_helpers_sbatch_check = None
_is_sbatch_in_path_error: Optional[Exception] = None
try:
    from .helpers import is_sbatch_in_path as _helpers_sbatch_check  # type: ignore
except Exception as exc:  # pragma: no cover - fallback when .helpers isn't importable
    _is_sbatch_in_path_error = exc


def _fallback_is_sbatch_in_path_inner() -> bool:
    return shutil.which("sbatch") is not None


def _public_is_sbatch_in_path() -> bool:
    if _helpers_sbatch_check is not None:
        return bool(_helpers_sbatch_check())
    if _is_sbatch_in_path_error is not None:
        # Silently fall back.  Callers can introspect
        # ``_is_sbatch_in_path_error`` if they want to surface the error.
        pass
    return _fallback_is_sbatch_in_path_inner()


# Module-level alias so callers can ``import executor_backend;
# executor_backend.is_sbatch_in_path()``.
is_sbatch_in_path = _public_is_sbatch_in_path


def is_slurm_env() -> bool:
    """Return True if we are currently running inside a SLURM allocation."""
    return "SLURM_JOB_ID" in os.environ


@dataclass
class BackendKind:
    """Identifies which submitit executor a :class:`Backend` uses."""

    name: str

    @property
    def is_local(self) -> bool:
        return self.name == "local"

    @property
    def is_auto(self) -> bool:
        return self.name == "auto"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def make_backend(
    folder: str,
    *,
    name: str,
    force_local: bool = False,
    timeout_min: int = 60,
    cpus_per_task: int = 1,
    mem_gb: int = 1,
    nodes: int = 1,
    gpus: int = 0,
    slurm_gres: Optional[str] = None,
    slurm_signal_delay_s: int = 0,
    slurm_use_srun: bool = False,
    exclude: str = "",
    stderr_to_stdout: bool = True,
    cluster: Optional[str] = None,
) -> "Backend":
    """Create a :class:`Backend` choosing ``AutoExecutor`` or ``LocalExecutor``.

    Parameters mirror what :class:`submitit.AutoExecutor.update_parameters`
    expects so the caller can stay agnostic about whether SLURM is available.
    """
    if submitit is None:
        raise RuntimeError(
            "submitit is not importable; install it with `pip install submitit`."
        )

    use_local = force_local or not is_sbatch_in_path()

    if use_local:
        executor: Any = LocalExecutor(folder=folder)
        kind = BackendKind("local")
    else:
        executor = AutoExecutor(folder=folder, cluster=cluster)
        kind = BackendKind("auto")

    params = {
        "name": name,
        "timeout_min": timeout_min,
        "cpus_per_task": cpus_per_task,
        "nodes": nodes,
        "mem_gb": mem_gb,
        "stderr_to_stdout": stderr_to_stdout,
    }

    if not use_local:
        # SLURM-only options
        params["slurm_gres"] = slurm_gres if slurm_gres is not None else f"gpu:{gpus}"
        params["slurm_signal_delay_s"] = slurm_signal_delay_s
        params["slurm_use_srun"] = slurm_use_srun
        params["exclude"] = exclude

    executor.update_parameters(**params)
    return Backend(executor=executor, kind=kind, folder=folder)


class Backend:
    """Thin wrapper around a submitit executor that records which kind we picked."""

    def __init__(self, executor: Any, kind: BackendKind, folder: str) -> None:
        self.executor = executor
        self.kind = kind
        self.folder = folder

    @property
    def is_local(self) -> bool:
        return self.kind.is_local

    @property
    def is_auto(self) -> bool:
        return self.kind.is_auto

    # ------------------------------------------------------------------
    # Submitit-compatible API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a callable to the underlying executor."""
        return self.executor.submit(fn, *args, **kwargs)

    def submit_array(self, fn: Callable[..., Any], *args_list: Iterable[Sequence[Any]]) -> List[Any]:
        """Submit a list of argument-tuples (for SLURM job arrays or local batch)."""
        jobs = []
        for args in args_list:
            jobs.append(self.executor.submit(fn, *args))
        return jobs

    def wait(self, jobs: Any, *, timeout: Optional[float] = None) -> List[Any]:
        return self.executor.wait(jobs, timeout=timeout)

    def map(self, fn: Callable[..., Any], iterable: Iterable[Sequence[Any]]) -> List[Any]:
        return [self.executor.submit(fn, *args) for args in iterable]


# ---------------------------------------------------------------------------
# CPU / RAM monitor with Rich progress
# ---------------------------------------------------------------------------

class CPUMonitor:
    """Periodically sample ``psutil`` and update a Rich progress bar.

    Usage::

        with CPUMonitor() as monitor:
            for job in jobs:
                monitor.tick(submitted=1)
                ...
            monitor.tick(submitted=0, completed=1)

    The progress bar has two sections:

    * top: number of submitted / completed jobs
    * bottom: current CPU% and RAM usage of the local machine
    """

    def __init__(
        self,
        *,
        console: Optional[Any] = None,
        total: int = 0,
        refresh_hz: float = 1.0,
    ) -> None:
        if Progress is None or Console is None:
            raise RuntimeError(
                "rich and psutil are required for CPUMonitor; "
                "install them with `pip install rich psutil`."
            )
        self.console = console or Console()
        self.refresh_hz = refresh_hz
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TextColumn("[green]CPU {task.fields[cpu]:>5.1f}%"),
            TextColumn("•"),
            TextColumn("[cyan]RAM {task.fields[ram]:>6.1f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=refresh_hz * 4,
            transient=False,
        )
        self._task = self.progress.add_task(
            "Workers",
            total=total if total > 0 else 1,
            completed=0,
            cpu=0.0,
            ram=0.0,
        )
        self._live: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        *,
        submitted: int = 0,
        completed: int = 0,
        failed: int = 0,
        total: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._submitted += submitted
            self._completed += completed
            self._failed += failed
            cpu, ram = _sample_cpu_ram()
            kwargs = {
                "completed": self._completed,
            }
            if total is not None:
                kwargs["total"] = total
            # ``rich.progress.update`` has a strict signature; cast the
            # kwargs to the right types so mypy stays happy.
            self.progress.update(
                self._task,
                completed=int(self._completed),
                total=int(kwargs["total"]) if total is not None else None,
                cpu=float(cpu),
                ram=float(ram),
            )

    def __enter__(self) -> "CPUMonitor":
        if Live is None:
            return self
        self._live = Live(self.progress, console=self.console, refresh_per_second=self.refresh_hz * 4)
        self._live.__enter__()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(1.0 / max(self.refresh_hz, 0.5)):
            cpu, ram = _sample_cpu_ram()
            with self._lock:
                self.progress.update(self._task, cpu=cpu, ram=ram)


def _sample_cpu_ram() -> tuple:
    """Return ``(cpu_percent, ram_percent)`` from psutil, or ``(0.0, 0.0)`` if missing."""
    if psutil is None:
        return (0.0, 0.0)
    try:
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().percent)
    except Exception:  # pragma: no cover - defensive
        return (0.0, 0.0)
    return cpu, ram


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test_executor() -> int:
    import tempfile

    failures = 0

    print(f"sbatch in path: {is_sbatch_in_path()}")
    print(f"slurm env:      {is_slurm_env()}")

    if submitit is None:
        print("submitit not installed, skipping executor tests")
    else:
        # Test the factory picks the right backend.
        with tempfile.TemporaryDirectory() as td:
            b = make_backend(td, name="test", timeout_min=5)
            print(f"backend kind:   {b.kind.name}")
            print(f"backend folder: {b.folder}")
            assert b.kind.name in ("local", "auto"), b.kind

            # Submit a small job (sleeps 0.1s) and wait for it.
            def _work(x: int) -> int:
                time.sleep(0.1)
                return x * 2

            jobs = b.submit_array(_work, [(i,) for i in range(4)])
            results = b.wait(jobs)
            if sorted(results) != [0, 2, 4, 6]:
                print(f"FAIL: expected [0,2,4,6], got {results}")
                failures += 1
            else:
                print("submit_array:  OK")

    # Test the CPU monitor.
    if Progress is not None and Console is not None and psutil is not None:
        try:
            with CPUMonitor(total=4) as mon:
                mon.tick(submitted=1)
                time.sleep(0.5)
                mon.tick(submitted=1, completed=1)
                time.sleep(0.5)
                mon.tick(submitted=0, completed=2)
                time.sleep(0.3)
            print("CPUMonitor:    OK")
        except Exception as exc:
            print(f"FAIL CPUMonitor: {exc}")
            failures += 1
    else:
        print("CPUMonitor:    skipped (rich/psutil missing)")

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_self_test_executor())
