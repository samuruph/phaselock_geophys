"""Contact sheets for eyeballing what the pipeline actually did.

Numbers hide things that a glance catches instantly: a clip resampled to the wrong frame
rate, a letterbox eating half the frame, an inversion that reconstructed something
plausible but *different*, a generation that is static. These write the frames out so a
run can be inspected rather than trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from . import palette


def sample_frames(frames: torch.Tensor, count: int = 8) -> tuple[torch.Tensor, list[int]]:
    """Evenly spaced frames, including the first and last."""
    total = frames.shape[0]
    if total <= count:
        return frames, list(range(total))
    index = torch.linspace(0, total - 1, count).round().long().tolist()
    return frames[index], index


def contact_sheet(
    rows: Mapping[str, torch.Tensor],
    path: Path,
    title: str = "",
    count: int = 8,
    note: str = "",
) -> Path:
    """One row of frames per named clip, sampled at the same time points.

    Rows share their frame indices, so a column is the same moment in every clip and
    differences read vertically.
    """
    import matplotlib.pyplot as plt

    palette.apply_style()
    if not rows:
        raise ValueError("no clips to draw")

    sampled = {name: sample_frames(frames, count) for name, frames in rows.items()}
    columns = max(len(index) for _, index in sampled.values())

    figure, axes = plt.subplots(
        len(sampled), columns,
        figsize=(1.55 * columns, 1.75 * len(sampled) + 0.9),
        squeeze=False, layout="constrained",
    )
    for row_index, (name, (frames, indices)) in enumerate(sampled.items()):
        for column in range(columns):
            axis = axes[row_index][column]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column < frames.shape[0]:
                axis.imshow(frames[column].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                if row_index == 0:
                    axis.set_title(f"frame {indices[column]}", fontsize=7.5,
                                   color=palette.TEXT_MUTED, loc="center")
            else:
                axis.axis("off")
        axes[row_index][0].set_ylabel(
            name, fontsize=8.5, color=palette.TEXT_SECONDARY, rotation=0,
            ha="right", va="center", labelpad=8,
        )

    if title:
        figure.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold",
                        color=palette.TEXT_PRIMARY)
    if note:
        figure.get_layout_engine().set(rect=(0, 0.045, 1, 0.96 if title else 1.0))
        palette.caption(figure, note)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return path


def inversion_sheet(
    original: torch.Tensor,
    vae_roundtrip: torch.Tensor,
    reconstructed: torch.Tensor,
    path: Path,
    sample_id: str = "",
    psnr_vae: Optional[float] = None,
    psnr_inversion: Optional[float] = None,
) -> Path:
    """Original, VAE round trip, and invert-then-resample, stacked for comparison.

    The middle row is the ceiling: no inversion can beat what the VAE alone preserves.
    Comparing the bottom row against it separates "the inversion lost this" from "the VAE
    never had it", which a single PSNR number cannot.
    """
    labels = {
        "original": original,
        f"VAE only{f'  {psnr_vae:.1f} dB' if psnr_vae else ''}": vae_roundtrip,
        f"inverted{f'  {psnr_inversion:.1f} dB' if psnr_inversion else ''}": reconstructed,
    }
    return contact_sheet(
        labels, path,
        title=f"Inversion fidelity — {sample_id}" if sample_id else "Inversion fidelity",
        note=(
            "Middle row is the ceiling: the VAE round trip alone, no inversion. The bottom "
            "row can only be worse. A difference between them is what inversion cost."
        ),
    )


def generation_sheet(
    reference: torch.Tensor,
    generated: Mapping[str, torch.Tensor],
    path: Path,
    sample_id: str = "",
) -> Path:
    """The real continuation on top, generations beneath it."""
    rows = {"real (reference)": reference}
    rows.update(generated)
    return contact_sheet(
        rows, path,
        title=f"Generated continuation — {sample_id}" if sample_id else "Generated continuation",
        note=(
            "All rows are conditioned on the same first frame, so column 0 should match. "
            "Divergence later is the model's own dynamics."
        ),
    )


def pair_sheet(
    plausible: torch.Tensor, violated: torch.Tensor, path: Path,
    scenario: str = "", violation: str = "",
) -> Path:
    """A matched pair side by side, to confirm the violation is actually visible.

    Worth checking before trusting any detection number: if temporal resampling stepped
    over the violation, the two rows look identical and no statistic can separate them.
    """
    return contact_sheet(
        {"plausible": plausible, f"violated ({violation})" if violation else "violated": violated},
        path,
        title=f"Matched pair — {scenario}" if scenario else "Matched pair",
        note=(
            "If these two rows look the same, the preprocessing has removed the violation "
            "and no statistic downstream can recover it."
        ),
    )
