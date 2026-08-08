"""Environment preflight, run before torch touches the GPU.

On this box a system CUDA install at ``/usr/local/cuda/lib64`` sits ahead of torch's
bundled ``site-packages/nvidia/*/lib`` on ``LD_LIBRARY_PATH``. Torch links against its own
``libcublasLt.so.13``, the loader hands it the system one instead, and the first cuBLASLt
call aborts the process:

    Invalid handle. Cannot load symbol cublasLtCreate

It is a hard crash, not a warning, and it happens deep inside the VAE encode rather than
at import, so it looks like a model bug rather than an environment one. Worse, a plain
``torch.matmul`` succeeds, so the obvious smoke test does not catch it.

``LD_LIBRARY_PATH`` is read by the dynamic loader at process start, so it cannot be fixed
from inside a running interpreter -- the only remedy is to fix the variable and re-exec.
:func:`ensure_cuda_libraries` does exactly that, once, guarded by an environment flag so
the re-executed process does not loop.

Call it at the top of an entry point, **before importing torch**.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GUARD = "PHASELOCK_RUNTIME_PREPARED"


def torch_nvidia_lib_dirs() -> list[str]:
    """The ``lib`` directories of torch's bundled NVIDIA wheels, if any.

    Located by walking site-packages rather than importing torch, so this stays usable
    before the interpreter has loaded it.
    """
    directories: list[str] = []
    for entry in sys.path:
        candidate = Path(entry) / "nvidia"
        if not candidate.is_dir():
            continue
        for package in sorted(candidate.iterdir()):
            lib = package / "lib"
            if lib.is_dir():
                directories.append(str(lib))
    return directories


def ensure_cuda_libraries(verbose: bool = True) -> bool:
    """Put torch's bundled CUDA libraries first on ``LD_LIBRARY_PATH``, re-execing if needed.

    Returns True if the process was already correctly configured. Otherwise it does not
    return -- the process is replaced.
    """
    if os.environ.get(GUARD) or sys.platform != "linux":
        return True

    directories = torch_nvidia_lib_dirs()
    if not directories:
        os.environ[GUARD] = "1"
        return True

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    # Already correct if every bundled directory precedes any system CUDA directory.
    system_cuda = [p for p in current if "/usr/local/cuda" in p or "/usr/lib" in p]
    first_system = min((current.index(p) for p in system_cuda), default=len(current))
    if all(d in current and current.index(d) < first_system for d in directories):
        os.environ[GUARD] = "1"
        return True

    ordered = directories + [p for p in current if p not in directories]
    os.environ["LD_LIBRARY_PATH"] = ":".join(ordered)
    os.environ[GUARD] = "1"

    if verbose:
        print(
            "[phaselock] putting torch's bundled CUDA libraries ahead of the system CUDA "
            "install and re-executing (see phaselock/runtime.py)",
            file=sys.stderr,
            flush=True,
        )
    os.execv(sys.executable, [sys.executable] + sys.argv)


def disable_core_dumps() -> None:
    """Stop a crash from stalling for minutes writing a multi-gigabyte core file.

    A segfaulting CogVideoX process holds tens of gigabytes; dumping that to disk makes a
    fast failure look like a hang, which is how the cuBLAS crash above was initially
    misdiagnosed.
    """
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:  # pragma: no cover - best effort only
        pass


def prepare(verbose: bool = True) -> None:
    """Everything an entry point should do before importing torch."""
    disable_core_dumps()
    ensure_cuda_libraries(verbose=verbose)
