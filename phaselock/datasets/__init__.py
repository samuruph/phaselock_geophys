"""Dataset registry."""

from __future__ import annotations

from typing import Any

from .base import (
    GenerationDataset,
    GenerationSample,
    PairedVideoDataset,
    VideoPair,
    VideoSample,
    balanced_subset,
)
from .intphys2 import IntPhys2
from .likephys import LikePhys
from .physics_iq import PhysicsIQ
from .video_io import (
    decode_video,
    gaussian_blur,
    letterbox,
    load_video,
    resample_frames,
    save_video,
    temporal_window,
)

PAIRED_DATASETS: dict[str, type[PairedVideoDataset]] = {
    "likephys": LikePhys,
    "intphys2": IntPhys2,
}

GENERATION_DATASETS: dict[str, type[GenerationDataset]] = {
    "physics_iq": PhysicsIQ,
}

DATASETS: dict[str, type] = {**PAIRED_DATASETS, **GENERATION_DATASETS}


def get_dataset(name: str, **kwargs: Any):
    """Instantiate a registered dataset by name."""
    try:
        cls = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; registered datasets are {sorted(DATASETS)}"
        ) from None
    return cls(**kwargs)


def get_paired_dataset(name: str, **kwargs: Any) -> PairedVideoDataset:
    """Instantiate a dataset that supports the paired detection protocol."""
    if name not in PAIRED_DATASETS:
        raise KeyError(
            f"{name!r} is not a paired dataset; paired datasets are {sorted(PAIRED_DATASETS)}"
        )
    return PAIRED_DATASETS[name](**kwargs)


__all__ = [
    "DATASETS",
    "GENERATION_DATASETS",
    "GenerationDataset",
    "GenerationSample",
    "IntPhys2",
    "LikePhys",
    "PAIRED_DATASETS",
    "PairedVideoDataset",
    "PhysicsIQ",
    "VideoPair",
    "VideoSample",
    "balanced_subset",
    "decode_video",
    "gaussian_blur",
    "get_dataset",
    "get_paired_dataset",
    "letterbox",
    "load_video",
    "resample_frames",
    "save_video",
    "temporal_window",
]
