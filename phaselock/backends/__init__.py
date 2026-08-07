"""Backend registry.

Adding a model means adding a :class:`~phaselock.backends.base.LatentSpec`, a
:class:`~phaselock.backends.base.VideoBackend` subclass, and one entry here.
Nothing downstream needs to know which model it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    DenoiserState,
    LatentSpec,
    VideoBackend,
    denormalize,
    from_canonical,
    normalize,
    num_latent_frames,
    to_canonical,
    token_grid,
)
from .cogvideox import COGVIDEOX_5B, CogVideoXBackend
from .wan import (
    WAN21_I2V_14B_480P,
    WAN21_I2V_14B_720P,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WanBackend,
)


@dataclass(frozen=True)
class BackendEntry:
    """Everything needed to instantiate a backend by name."""

    backend_cls: type[VideoBackend]
    spec: LatentSpec
    default_model_id: str
    mode: str
    validated: bool = True
    """False for checkpoints too large to run on the development box.

    Layout, normalisation and conditioning are implemented for these, but they have
    not been exercised end to end.
    """


BACKENDS: dict[str, BackendEntry] = {
    "cogvideox_5b_t2v": BackendEntry(
        CogVideoXBackend, COGVIDEOX_5B, "/data/weights/CogVideoX-5b-Diffusers", "t2v"
    ),
    "cogvideox_5b_i2v": BackendEntry(
        CogVideoXBackend, COGVIDEOX_5B, "THUDM/CogVideoX-5B-I2V", "i2v"
    ),
    "wan21_t2v_1_3b": BackendEntry(
        WanBackend, WAN21_T2V_1_3B, "/data/weights/Wan2.1-T2V-1.3B-Diffusers", "t2v"
    ),
    "wan21_t2v_14b": BackendEntry(
        WanBackend, WAN21_T2V_14B, "/data/weights/Wan2.1-T2V-14B-Diffusers", "t2v", validated=False
    ),
    "wan21_i2v_14b_480p": BackendEntry(
        WanBackend, WAN21_I2V_14B_480P, "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "i2v", validated=False
    ),
    "wan21_i2v_14b_720p": BackendEntry(
        WanBackend, WAN21_I2V_14B_720P, "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", "i2v", validated=False
    ),
}


def get_entry(name: str) -> BackendEntry:
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered backends are {sorted(BACKENDS)}"
        ) from None


def get_spec(name: str) -> LatentSpec:
    """The latent spec for a backend, without loading any weights."""
    return get_entry(name).spec


def load_backend(name: str, model_id: str | None = None, **kwargs: Any) -> VideoBackend:
    """Instantiate a registered backend, downloading or loading weights."""
    entry = get_entry(name)
    return entry.backend_cls.from_pretrained(
        model_id or entry.default_model_id,
        spec=entry.spec,
        mode=entry.mode,
        **kwargs,
    )


__all__ = [
    "BACKENDS",
    "BackendEntry",
    "COGVIDEOX_5B",
    "CogVideoXBackend",
    "DenoiserState",
    "LatentSpec",
    "VideoBackend",
    "WAN21_I2V_14B_480P",
    "WAN21_I2V_14B_720P",
    "WAN21_T2V_14B",
    "WAN21_T2V_1_3B",
    "WanBackend",
    "denormalize",
    "from_canonical",
    "get_entry",
    "get_spec",
    "load_backend",
    "normalize",
    "num_latent_frames",
    "to_canonical",
    "token_grid",
]
