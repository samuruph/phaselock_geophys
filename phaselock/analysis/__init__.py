"""Figures and styling for the results.

Kept apart from the metrics so that plotting never becomes a dependency of measurement,
and so a figure can be re-rendered from a CSV without a GPU.
"""

from . import palette, visuals
from .figures import (
    depth_profile,
    depth_time_heatmaps,
    drift_comparison,
    latent_motion_profile,
    render_all,
    signal_profile,
    source_comparison,
    statistic_comparison,
    step_sweep,
    step_sweep_panels,
    generation_quality,
)

__all__ = [
    "depth_profile",
    "depth_time_heatmaps",
    "drift_comparison",
    "latent_motion_profile",
    "palette",
    "visuals",
    "render_all",
    "signal_profile",
    "source_comparison",
    "statistic_comparison",
    "step_sweep",
    "step_sweep_panels",
    "generation_quality",
]
