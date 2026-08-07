"""Frozen DINOv2 features -- the external GeoPhys path.

This is the correctness gate for the whole project. GeoPhys reports 78-81% pairwise
accuracy on LikePhys from a single frozen backbone; if the five statistics computed on
DINOv2 features do not land near that, the implementation is wrong and no number measured
on internal representations means anything.

The paper selects a readout layer per backbone by maximising the curvature gap between
plausible and violated clips on a held-out split, and reports L12 of DINOv2 ViT-L/24.
:meth:`DINOv2Encoder.select_layer` reproduces that procedure rather than hardcoding the
index, since the layer that wins depends on the data.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

DEFAULT_MODEL = "facebook/dinov2-large"

# ImageNet statistics, which DINOv2 was trained with.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DINOv2Encoder:
    """Per-frame DINOv2 features, spatially pooled into a trajectory.

    Args:
        model_id: HuggingFace id. ``facebook/dinov2-large`` is the ViT-L/14 the paper used.
        layer: Which hidden layer to read out. ``None`` uses the final layer; the paper's
            DINOv2 choice is 12 of 24.
        image_size: Frames are resized to this before encoding. 224 keeps the patch grid
            at 16x16 and the cost low.
        device / dtype: Where and how to run.
    """

    name = "dinov2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        layer: Optional[int] = None,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 16,
    ):
        from transformers import AutoModel

        self.model_id = model_id
        self.layer = layer
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.num_layers = self.model.config.num_hidden_layers
        # DINOv2 prepends a CLS token; the "with registers" variants prepend more. Those
        # are not spatial, so pooling them in would contaminate every frame equally and
        # dilute the motion signal.
        self.num_prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """``(F, 3, H, W)`` in [0, 1] -> resized and ImageNet-normalised."""
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected (F, 3, H, W) frames, got {tuple(frames.shape)}")
        resized = torch.nn.functional.interpolate(
            frames, size=(self.image_size, self.image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        mean = torch.tensor(_MEAN, device=resized.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=resized.device).view(1, 3, 1, 1)
        return (resized - mean) / std

    @torch.no_grad()
    def encode(self, frames: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """Frames -> per-frame trajectory ``(F, D)``.

        Spatial average pooling over patch tokens, matching the paper's
        ``z_t = (1/N) sum_n z_{t,n}``.
        """
        layer = self.layer if layer is None else layer
        pixels = self._preprocess(frames.to(self.device))

        pooled: list[torch.Tensor] = []
        for start in range(0, pixels.shape[0], self.batch_size):
            batch = pixels[start : start + self.batch_size].to(self.model.dtype)
            outputs = self.model(pixel_values=batch, output_hidden_states=layer is not None)
            if layer is None:
                tokens = outputs.last_hidden_state
            else:
                if not 0 <= layer <= self.num_layers:
                    raise ValueError(
                        f"layer must be in [0, {self.num_layers}], got {layer}"
                    )
                # hidden_states[0] is the embedding output, so index i is "after i blocks".
                tokens = outputs.hidden_states[layer]
            pooled.append(tokens[:, self.num_prefix_tokens :].float().mean(dim=1).cpu())

        return torch.cat(pooled)

    @torch.no_grad()
    def encode_all_layers(self, frames: torch.Tensor) -> dict[int, torch.Tensor]:
        """Trajectories at every layer, for the readout-selection sweep.

        One forward pass produces all of them, so a layer sweep costs no more compute
        than a single-layer run.
        """
        pixels = self._preprocess(frames.to(self.device))
        per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(self.num_layers + 1)}

        for start in range(0, pixels.shape[0], self.batch_size):
            batch = pixels[start : start + self.batch_size].to(self.model.dtype)
            outputs = self.model(pixel_values=batch, output_hidden_states=True)
            for index, tokens in enumerate(outputs.hidden_states):
                per_layer[index].append(
                    tokens[:, self.num_prefix_tokens :].float().mean(dim=1).cpu()
                )

        return {index: torch.cat(parts) for index, parts in per_layer.items()}


def select_layer(
    plausible_trajectories: Sequence[dict[int, torch.Tensor]],
    violated_trajectories: Sequence[dict[int, torch.Tensor]],
    statistic: str = "curv",
    order: int = 3,
) -> tuple[int, dict[int, float]]:
    """Pick the readout layer that maximises the violated-minus-plausible gap.

    The paper selects ``l*`` by maximising the curvature delta on a held-out split, and
    reports that the geometric and task-driven criteria coincide -- the same layer also
    maximises detection accuracy.

    Args:
        plausible_trajectories: One ``{layer: (F, D)}`` mapping per plausible clip, as
            returned by :meth:`DINOv2Encoder.encode_all_layers`.
        violated_trajectories: The matched violated clips, in the same order.
        statistic: Which of the five to maximise the gap on.

    Returns:
        ``(best_layer, {layer: gap})``.
    """
    from ..metrics.geophys import geophys_statistics

    if len(plausible_trajectories) != len(violated_trajectories):
        raise ValueError("plausible and violated trajectory lists must be the same length")
    if not plausible_trajectories:
        raise ValueError("no trajectories supplied")

    layers = sorted(plausible_trajectories[0])
    gaps: dict[int, float] = {}
    for layer in layers:
        deltas = [
            float(geophys_statistics(bad[layer], order=order)[statistic])
            - float(geophys_statistics(good[layer], order=order)[statistic])
            for good, bad in zip(plausible_trajectories, violated_trajectories)
        ]
        gaps[layer] = sum(deltas) / len(deltas)

    return max(gaps, key=gaps.__getitem__), gaps
