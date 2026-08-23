"""Safe process-pool helper for macOS.

macOS uses the ``spawn`` start method.  A spawned child re-imports the parent's
``__main__`` module under the name ``__mp_main__``.  If that module's body calls
into a pool itself -- i.e. the script has no ``if __name__ == "__main__":``
guard -- the child raises inside ``_check_not_importing_main`` and the parent
blocks forever in ``pool.map``.  ``forkserver`` does not help: the fork server is
itself started by spawn and re-imports main the same way.

Rather than making every caller responsible for the guard (and hanging silently
when they forget), :func:`parallel_map` checks up front and degrades to serial
execution with a clear warning.  Collection and CMA-ES evaluation are both
embarrassingly parallel but both correct serially, so degrading is always safe.
"""
from __future__ import annotations

import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from wmcore.utils.logging_utils import get_logger

log = get_logger(__name__)

_GUARD_RE = re.compile(r"__name__\s*==\s*['\"]__main__['\"]")


def spawn_is_safe() -> tuple[bool, str]:
    """Can a spawned child safely re-import this process's ``__main__``?

    Three cases:

    * no ``__main__.__file__`` (``python -c``, ``python -``, a REPL, a notebook)
      -- nothing is re-imported, so spawning is safe;
    * ``__main__`` is a file containing an ``if __name__ == "__main__":`` guard
      -- the body is skipped on re-import, safe;
    * ``__main__`` is a file without one -- the body re-runs in the child, unsafe.

    The guard check is a source-level heuristic, deliberately so: there is no
    runtime API that answers this question, and a false negative only costs
    parallelism, never correctness.
    """
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if main_file is None:
        return True, "no __main__ file to re-import"
    try:
        source = Path(main_file).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False, f"could not read {main_file}"
    if _GUARD_RE.search(source):
        return True, "__main__ guard present"
    return False, f"{Path(main_file).name} has no `if __name__ == \"__main__\":` guard"


def parallel_map(
    fn: Callable[[Any], Any],
    jobs: Sequence[Any],
    *,
    n_workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    label: str = "work",
) -> list[Any]:
    """``pool.map`` with an automatic serial fallback.

    ``fn`` must be importable by name in a fresh interpreter (a module-level
    function, not a closure or a lambda).
    """
    jobs = list(jobs)
    n_workers = max(1, min(int(n_workers), len(jobs)))

    safe, reason = spawn_is_safe()
    if n_workers == 1 or not safe:
        if not safe:
            log.warning(
                "running %s serially: %s. For parallel execution use "
                "`python wm.py ...` or add a `if __name__ == \"__main__\":` "
                "guard to your script.", label, reason)
        if initializer is not None:
            initializer(*initargs)
        return [fn(job) for job in jobs]

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        return pool.map(fn, jobs)


class ProcessMapper:
    """A reusable worker pool with the same serial fallback as :func:`parallel_map`.

    CMA-ES calls ``map`` once per generation and its workers each hold a loaded
    copy of V and M.  Creating a fresh pool per generation would reload both
    models in every worker every generation, which on a 40-generation run costs
    more than the search itself.  This keeps one pool for the whole run.

    Use as a context manager::

        with ProcessMapper(6, initializer=_init, initargs=(cfg,)) as mapper:
            for gen in range(n):
                results = mapper.map(_evaluate, jobs)
    """

    def __init__(self, n_workers: int, *, initializer: Callable[..., None] | None = None,
                 initargs: tuple = (), label: str = "work"):
        self.n_workers = max(1, int(n_workers))
        self.initializer = initializer
        self.initargs = initargs
        self.label = label
        self._pool = None
        self.parallel = False

    def __enter__(self) -> "ProcessMapper":
        safe, reason = spawn_is_safe()
        if self.n_workers > 1 and safe:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self.n_workers, initializer=self.initializer,
                                  initargs=self.initargs)
            self.parallel = True
        else:
            if not safe:
                log.warning("running %s serially: %s.", self.label, reason)
            if self.initializer is not None:
                self.initializer(*self.initargs)
        return self

    def map(self, fn: Callable[[Any], Any], jobs: Iterable[Any]) -> list[Any]:
        jobs = list(jobs)
        if self._pool is not None:
            return self._pool.map(fn, jobs)
        return [fn(job) for job in jobs]

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
