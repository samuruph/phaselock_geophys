"""Small shared helpers.

Video encoding used to live here; it now belongs to the backends, which know their own
latent layout and normalisation convention. See :meth:`phaselock.backends.base.VideoBackend.encode`.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    """Map a config string to a torch dtype."""
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(dtypes)}") from None


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Shape and moments, for logging and debugging."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(tensor.float().mean()),
        "std": float(tensor.float().std()),
        "min": float(tensor.float().min()),
        "max": float(tensor.float().max()),
    }
