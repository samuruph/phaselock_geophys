"""A progress bar for the long drivers.

Every stage is a loop over clips at 1.8 to 5.5 minutes each, so a run is hours and the
only feedback was a log line per clip. That tells you where you are but not when you will
be done, which is the thing you actually want when deciding whether to wait.

Wrapped rather than used directly so the drivers stay readable, the bar can be turned off
for a non-interactive log without touching them, and the rate is reported in the units the
work is actually measured in.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

DISABLE_ENV = "PHASELOCK_NO_PROGRESS"
"""Set to any non-empty value to fall back to plain logging."""


def track(
    items: Iterable[T],
    description: str,
    total: Optional[int] = None,
    unit: str = "clip",
) -> Iterator[T]:
    """Iterate with a progress bar, or plainly if tqdm is unavailable or disabled.

    Writes to stderr so a redirected stdout keeps only the run's own output, and leaves
    the finished bar in place so a log read afterwards still shows how long the stage
    took.
    """
    if os.environ.get(DISABLE_ENV):
        yield from items
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is in requirements
        yield from items
        return

    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None

    yield from tqdm(
        items,
        desc=description,
        total=total,
        unit=unit,
        file=sys.stderr,
        dynamic_ncols=True,
        # Show elapsed and remaining: the whole point is answering "when is this done".
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True,
    )
