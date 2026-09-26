"""Focused, opt-in running-momentum diagnostics and synchronized video rendering."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from .video import encode_frames


_EPS = 1e-8


def _cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.detach().float().cpu()


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt())


def _temporal_rms(x: torch.Tensor) -> torch.Tensor:
    """One channel-and-space RMS value per latent-frame transition."""
    return x.float().square().mean(dim=(1, 2, 3)).sqrt()


def _sqrt_m2_temporal_rms(m2: torch.Tensor) -> torch.Tensor:
    return m2.float().mean(dim=(1, 2, 3)).clamp_min(0).sqrt()


def _mean_signal(step: "DiagnosticStep") -> torch.Tensor:
    return step.m1 if step.mean is None else step.mean


def _variance_signal(step: "DiagnosticStep") -> torch.Tensor:
    return step.m2 if step.variance is None else step.variance


@dataclass
class DiagnosticStep:
    step: int
    timestep: float
    tau: float
    current: torch.Tensor
    m1: torch.Tensor
    m2: torch.Tensor
    correction: torch.Tensor
    mean: Optional[torch.Tensor] = None
    variance: Optional[torch.Tensor] = None
    x0: Optional[torch.Tensor] = None
    latents_before: Optional[torch.Tensor] = None
    latents_after: Optional[torch.Tensor] = None
    flow: Optional[torch.Tensor] = None
    strength: float = 0.0
    latent_weight: Optional[float] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        applied = self.correction * self.strength
        mean = self.m1 if self.mean is None else self.mean
        variance = self.m2 if self.variance is None else self.variance
        result = {
            "step": self.step,
            "timestep": self.timestep,
            "tau": self.tau,
            "lambda": self.strength,
            "latent_weight": self.latent_weight,
            "current_rms": _rms(self.current),
            "bias_corrected_mean_rms": _rms(mean),
            "bias_corrected_variance_mean": float(variance.float().mean()),
            "sqrt_bias_corrected_variance_rms": float(variance.float().mean().clamp_min(0).sqrt()),
            "raw_m1_rms": _rms(self.m1),
            "raw_m2_mean": float(self.m2.float().mean()),
            # Compatibility aliases now report the bias-corrected signals shown in the dashboard.
            "m1_rms": _rms(mean),
            "m2_mean": float(variance.float().mean()),
            "sqrt_m2_rms": float(variance.float().mean().clamp_min(0).sqrt()),
            "applied_guidance_rms": _rms(applied),
            "current_by_latent_transition": _temporal_rms(self.current).tolist(),
            "bias_corrected_mean_by_latent_transition": _temporal_rms(mean).tolist(),
            "bias_corrected_variance_mean_by_latent_transition": variance.float().mean(dim=(1, 2, 3)).tolist(),
            "sqrt_bias_corrected_variance_rms_by_latent_transition": _sqrt_m2_temporal_rms(variance).tolist(),
            "m1_by_latent_transition": _temporal_rms(mean).tolist(),
            "m2_mean_by_latent_transition": variance.float().mean(dim=(1, 2, 3)).tolist(),
            "sqrt_m2_rms_by_latent_transition": _sqrt_m2_temporal_rms(variance).tolist(),
            "applied_guidance_by_latent_transition": _temporal_rms(applied).tolist(),
        }
        if self.extras:
            result["extensions"] = {
                key: ({"rms": _rms(value), "min": float(value.min()), "max": float(value.max())}
                      if torch.is_tensor(value) else value)
                for key, value in self.extras.items()
            }
        return result


@dataclass
class MomentumTrace:
    """Per-step moment and applied-guidance tensors, with optional raw persistence."""

    name: str
    save_raw: bool = False
    record_steps: Optional[set[int]] = None
    steps: list[DiagnosticStep] = field(default_factory=list)

    def capture(
        self, *, step: int, timestep: torch.Tensor | float, tau: float,
        current: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor,
        correction: torch.Tensor, strength: float, x0: Optional[torch.Tensor] = None,
        latents_before: Optional[torch.Tensor] = None,
        latents_after: Optional[torch.Tensor] = None, flow: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None, variance: Optional[torch.Tensor] = None,
        latent_weight: Optional[float] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.record_steps is not None and step not in self.record_steps:
            return
        self.steps.append(DiagnosticStep(
            step=step,
            timestep=float(timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep),
            tau=float(tau), current=_cpu(current), m1=_cpu(m1), m2=_cpu(m2),
            correction=_cpu(correction), mean=_cpu(mean), variance=_cpu(variance),
            strength=float(strength), x0=_cpu(x0),
            latent_weight=latent_weight,
            latents_before=_cpu(latents_before) if self.save_raw else None,
            latents_after=_cpu(latents_after) if self.save_raw else None, flow=_cpu(flow),
            extras={k: (_cpu(v) if torch.is_tensor(v) else v)
                    for k, v in (extras or {}).items() if self.save_raw or k != "inner_tensors"},
        ))

    def summaries(self) -> list[dict[str, Any]]:
        return [step.summary() for step in self.steps]

    def save(self, directory: str | Path, provenance: Optional[dict[str, Any]] = None) -> Path:
        """Save full tensors only when explicitly requested; summary is written by renderer."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.save_raw:
            torch.save({"steps": self.steps, "provenance": provenance or {}},
                       directory / "raw_trace.pt")
        return directory


def _magnitude_map(field: torch.Tensor) -> torch.Tensor:
    return field.float().square().mean(dim=0).sqrt()


def _display_scales(trace: MomentumTrace) -> dict[str, float]:
    grouped: dict[str, list[torch.Tensor]] = {
        "current": [], "m1": [], "m2": [], "guidance": []}
    for step in trace.steps:
        grouped["current"].append(step.current.float().square().mean(dim=1).sqrt().flatten())
        grouped["m1"].append(_mean_signal(step).float().square().mean(dim=1).sqrt().flatten())
        grouped["m2"].append(_variance_signal(step).float().mean(dim=1).flatten())
        grouped["guidance"].append((step.correction.float() * step.strength)
                                    .square().mean(dim=1).sqrt().flatten())
    return {
        name: max(float(torch.quantile(torch.cat(values), .99)), _EPS)
        for name, values in grouped.items()
    }


def _map(
    field: torch.Tensor, scale: float, size: tuple[int, int], *, m2: bool = False,
) -> np.ndarray:
    values = field.float().mean(dim=0) if m2 else _magnitude_map(field)
    normalized = np.clip(values.numpy() / max(scale, _EPS), 0, 1)
    if m2:
        normalized = np.log1p(9 * normalized) / np.log(10)
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def _caption(image: np.ndarray, title: str) -> np.ndarray:
    height, width = image.shape[:2]
    band = np.full((28, width, 3), 24, np.uint8)
    scale = min(.48, max(.30, (width - 12) / max(len(title) * 9.5, 1)))
    cv2.putText(band, title, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate((band, image), axis=0)


def _chart(
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str, width: int, height: int, x_values: np.ndarray,
    marker_x: float, x_label: str, x_range: tuple[float, float],
    *, fixed_y_max: Optional[float] = None,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, np.uint8)
    left, right, top, bottom = 54, width - 14, 44, height - 34
    cv2.putText(canvas, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .49,
                (245, 245, 245), 1, cv2.LINE_AA)
    clean = [(n, np.asarray(v, np.float64), c) for n, v, c in series]
    valid_sets = [v[np.isfinite(v)] for _, v, _ in clean if np.isfinite(v).any()]
    all_values = np.concatenate(valid_sets) if valid_sets else np.array([1.])
    ymax = max(fixed_y_max or float(all_values.max()), _EPS)
    x0, x1 = x_range
    if x1 <= x0:
        x1 = x0 + 1.
    for fraction in (0., .5, 1.):
        y = int(bottom - fraction * (bottom-top))
        cv2.line(canvas, (left, y), (right, y), (65, 65, 65), 1)
        cv2.putText(canvas, f"{fraction*ymax:.2g}", (3, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, .30, (190, 190, 190), 1)
    legend_x = left
    for name, values, color in clean:
        valid = np.isfinite(values) & np.isfinite(x_values)
        if not valid.any():
            continue
        x = left + np.clip((x_values[valid]-x0)/(x1-x0), 0, 1) * (right-left)
        y = bottom - np.clip(values[valid]/ymax, 0, 1) * (bottom-top)
        points = np.stack((x, y), axis=1).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        marker_index = int(np.argmin(np.abs(x_values[valid]-marker_x)))
        cv2.circle(canvas, tuple(points[marker_index]), 4, color, -1)
        cv2.putText(canvas, name, (legend_x, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, color, 1, cv2.LINE_AA)
        legend_x += max(105, 12 + len(name)*7)
    marker = int(left + np.clip((marker_x-x0)/(x1-x0), 0, 1)*(right-left))
    cv2.line(canvas, (marker, top), (marker, bottom), (230, 230, 230), 1)
    cv2.putText(canvas, f"{x0:g}", (left, height-17), cv2.FONT_HERSHEY_SIMPLEX,
                .32, (190, 190, 190), 1)
    cv2.putText(canvas, f"{x1:g}", (right-24, height-17), cv2.FONT_HERSHEY_SIMPLEX,
                .32, (190, 190, 190), 1)
    cv2.putText(canvas, x_label, (width//2-48, height-3), cv2.FONT_HERSHEY_SIMPLEX,
                .34, (210, 210, 210), 1)
    return canvas


def _decode_clean(
    x0: Optional[torch.Tensor], decode: Optional[Callable[[torch.Tensor], torch.Tensor]],
    height: int, expected_frames: int, size: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if x0 is None or decode is None:
        width = size[0] if size else height
        return np.zeros((expected_frames, height, width, 3), np.uint8), (width, height)
    frames = decode(x0)
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("decoder must return (F, 3, H, W) RGB frames")
    source_height, source_width = frames.shape[-2:]
    width, height = size or (int(round(height * source_width/source_height)), height)
    result = np.stack([
        cv2.resize((frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255)
                   .astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ])
    return result, (width, height)


def _video_to_transition(frame: int, ratio: int, transition_count: int) -> Optional[int]:
    """Map decoded frame to the latent transition that spans its time interval."""
    if frame == 0:
        return None
    return min((frame - 1) // ratio, transition_count - 1)


def render_dashboard(
    guided: MomentumTrace, path: str | Path, *,
    decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    fps: int = 8, preview_height: int = 180,
    provenance: Optional[dict[str, Any]] = None, temporal_ratio: int = 4,
) -> Path:
    """Write one combined dashboard and one full temporal movie per recorded step."""
    if not guided.steps:
        raise ValueError("no diagnostic steps to render")
    if temporal_ratio < 1:
        raise ValueError("temporal_ratio must be positive")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = (provenance or {}).get("running_momentum_source", "latent")
    current_title = ("x0_hat delta at t" if source == "x0_hat" else
                     "Blended motion delta" if source == "blend" else
                     "Post-step latent delta")
    expected_frames = 1 + temporal_ratio * (guided.steps[0].current.shape[0] + 1)
    videos: list[np.ndarray] = []
    panel_size = None
    for step in guided.steps:
        video, panel_size = _decode_clean(step.x0, decode, preview_height, expected_frames, panel_size)
        videos.append(video)
    panel_width, panel_height = panel_size
    panel_gutter = 4
    dashboard_width = panel_width * 5 + panel_gutter * 4
    chart_height = max(170, preview_height)
    scales = _display_scales(guided)
    summaries = guided.summaries()
    diffusion_x = np.asarray([s.tau for s in guided.steps], np.float64)
    observed_strength_max = max((s.strength for s in guided.steps), default=0.)
    strength_max = observed_strength_max if observed_strength_max > 0 else 1.
    magnitudes = [
        ("Current delta", np.asarray([_rms(s.current) for s in guided.steps]), (0, 165, 255)),
        ("Corrected mean", np.asarray([_rms(_mean_signal(s)) for s in guided.steps]), (255, 255, 0)),
        ("sqrt(var)", np.asarray([
            _rms(_variance_signal(s).float().clamp_min(0).sqrt()) for s in guided.steps
        ]), (180, 80, 220)),
        ("Applied guide", np.asarray([_rms(s.correction*s.strength)
                                      for s in guided.steps]), (80, 210, 100)),
    ]
    schedule = [("lambda", np.asarray([s.strength for s in guided.steps]), (100, 190, 255))]
    tau_range = (0., 1.)
    colors = {name: color for name, _, color in magnitudes}
    moment_denoise_max = max(
        max((_rms(s.current) for s in guided.steps), default=0.),
        max((_rms(_mean_signal(s)) for s in guided.steps), default=0.),
        max((_rms(_variance_signal(s).float().clamp_min(0).sqrt()) for s in guided.steps), default=0.), _EPS)
    guidance_denoise_max = max((float(v) for name, values, _color in magnitudes
                                if name == "Applied guide" for v in values), default=_EPS)
    video_time_scales = {
        step.step: max(
            float(_temporal_rms(step.current).max()),
            float(_temporal_rms(_mean_signal(step)).max()),
            float(_sqrt_m2_temporal_rms(_variance_signal(step)).max()),
            _EPS,
        )
        for step in guided.steps
    }
    guidance_time_max = max((float(_temporal_rms(s.correction*s.strength).max())
                             for s in guided.steps), default=_EPS)

    def render_frames():
        for step_index, step in enumerate(guided.steps):
            video = videos[step_index]
            frame_count = len(video)
            frame_x = np.arange(frame_count, dtype=np.float64)
            time_values = {name: np.full(frame_count, np.nan) for name in colors}
            current_profile = _temporal_rms(step.current).numpy()
            mean_signal = _mean_signal(step)
            variance_signal = _variance_signal(step)
            m1_profile = _temporal_rms(mean_signal).numpy()
            m2_profile = _sqrt_m2_temporal_rms(variance_signal).numpy()
            guidance_profile = _temporal_rms(step.correction*step.strength).numpy()
            for frame in range(1, frame_count):
                transition = _video_to_transition(frame, temporal_ratio, len(m1_profile))
                time_values["Current delta"][frame] = current_profile[transition]
                time_values["Corrected mean"][frame] = m1_profile[transition]
                time_values["sqrt(var)"][frame] = m2_profile[transition]
                time_values["Applied guide"][frame] = guidance_profile[transition]
            temporal_series = [(name, time_values[name], colors[name]) for name in colors]
            denoise_moments = magnitudes[:3]
            denoise_guidance = [magnitudes[3]]
            plot_widths = [dashboard_width//3, dashboard_width//3,
                           dashboard_width - 2*(dashboard_width//3)]
            diffusion_moments_plot = _chart(
                denoise_moments, "Moments | diffusion time",
                plot_widths[0], chart_height, diffusion_x, step.tau,
                "tau: noise -> clean", tau_range, fixed_y_max=moment_denoise_max)
            diffusion_guidance_plot = _chart(
                denoise_guidance, "Applied guidance | diffusion time",
                plot_widths[1], chart_height, diffusion_x, step.tau,
                "tau: noise -> clean", tau_range, fixed_y_max=guidance_denoise_max)
            schedule_plot = _chart(
                schedule, "Schedule lambda | diffusion time",
                plot_widths[2], chart_height, diffusion_x, step.tau,
                "tau: noise -> clean", tau_range, fixed_y_max=strength_max)
            def compose(frame_index: int) -> np.ndarray:
                clean = video[frame_index]
                transition = _video_to_transition(frame_index, temporal_ratio, mean_signal.shape[0])
                if transition is None:
                    current_image = m1_image = m2_image = guidance_image = np.zeros_like(clean)
                    transition_text = "anchor frame; no prior transition"
                else:
                    current_image = _map(step.current[transition], scales["current"], panel_size)
                    m1_image = _map(mean_signal[transition], scales["m1"], panel_size)
                    m2_image = _map(variance_signal[transition], scales["m2"], panel_size, m2=True)
                    guidance_image = _map(step.correction[transition]*step.strength,
                                          scales["guidance"], panel_size)
                    transition_text = f"latent transition {transition+1}/{mean_signal.shape[0]}"
                header = np.full((40, dashboard_width, 3), 18, np.uint8)
                title = (f"Denoising step index {step.step} | t={step.timestep:.0f}"
                         f" | tau={step.tau:.3f} | lambda={step.strength:.3g}"
                         + (f" | latent weight={step.latent_weight:.2f}"
                            if step.latent_weight is not None else "")
                         + f" | frame {frame_index+1}/{frame_count} | {transition_text}")
                cv2.putText(header, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .42,
                            (250, 250, 250), 1, cv2.LINE_AA)
                panels = [
                    _caption(clean, "Predicted clean video"),
                    _caption(current_image, f"{current_title} | RMS"),
                    _caption(m1_image, "Bias-corrected mean | RMS"),
                    _caption(m2_image, "Corrected variance"),
                    _caption(guidance_image, "Applied guidance | RMS"),
                ]
                gutter = np.zeros((panels[0].shape[0], panel_gutter, 3), np.uint8)
                pieces = []
                for index, panel in enumerate(panels):
                    if index:
                        pieces.append(gutter)
                    pieces.append(panel)
                image_row = np.concatenate(pieces, axis=1)
                lower = np.concatenate((diffusion_moments_plot,
                                        diffusion_guidance_plot, schedule_plot), axis=1)
                temporal_moments_plot = _chart(
                    temporal_series[:3], "Current delta and corrected moments | video time",
                    dashboard_width//2, chart_height, frame_x, float(frame_index),
                    "decoded video frame", (0., float(max(frame_count-1, 1))),
                    fixed_y_max=video_time_scales[step.step])
                temporal_guidance_plot = _chart(
                    [temporal_series[3]], "Applied guidance | video time",
                    dashboard_width-dashboard_width//2, chart_height, frame_x, float(frame_index),
                    "decoded video frame", (0., float(max(frame_count-1, 1))),
                    fixed_y_max=guidance_time_max)
                canvas = np.concatenate((header, image_row, lower,
                                         np.concatenate((temporal_moments_plot,
                                                         temporal_guidance_plot), axis=1)), axis=0)
                if canvas.shape[0] % 2 or canvas.shape[1] % 2:
                    canvas = np.pad(canvas, ((0, canvas.shape[0] % 2),
                                             (0, canvas.shape[1] % 2), (0, 0)))
                return canvas

            def step_frames():
                for frame_index in range(frame_count):
                    yield compose(frame_index)

            step_path = out.parent / "steps" / f"step_{step.step:03d}.mp4"
            step_path.parent.mkdir(parents=True, exist_ok=True)
            encode_frames(step_frames(), step_path, fps=fps)
            for frame_index in range(frame_count):
                yield compose(frame_index)

    steps_directory = out.parent / "steps"
    if steps_directory.exists():
        shutil.rmtree(steps_directory)
    encode_frames(render_frames(), out, fps=fps)

    payload = {
        "schema_version": 3,
        "provenance": provenance or {},
        "alignment": {
            "latent_to_video_temporal_ratio": temporal_ratio,
            "map_rule": "latent transition j is held on decoded frames 1+j*r through min((j+1)*r, last frame); decoded frame 0 is the anchor",
            "spatial_rule": "each latent map is resized to the decoded video frame dimensions without changing its aspect ratio",
        },
        "signal_definitions": {
            "current_delta": ("adjacent-frame difference of x0_hat(z_t, t) from the current model prediction"
                              if source == "x0_hat" else
                              "smooth weighted blend of current-step x0_hat and post-step latent adjacent-frame differences"
                              if source == "blend" else
                              "adjacent-frame difference of callback latents after the scheduler update")
                             + "; panel and chart show RMS across latent channels",
            "m1": "bias-corrected running mean used by the controller, visualized as per-pixel RMS across latent channels; raw_m1 is also included in each step summary",
            "m2": "bias-corrected running variance used by the controller, visualized as per-pixel channel mean; plot uses sqrt(variance) in latent units; raw_m2 is also included in each step summary",
            "applied_guidance": "schedule strength lambda multiplied by the controller correction, visualized as per-pixel RMS across latent channels",
            "diffusion_time": "tau = 0 at noise and tau = 1 at clean data",
        },
        "display_scales_p99": scales,
        "video_time_plot_scale_by_step": {
            str(step): {"y_max": scale, "shared_by": ["current_delta", "corrected_mean", "sqrt_corrected_variance"]}
            for step, scale in video_time_scales.items()
        },
        "steps": summaries,
        "step_videos": [f"steps/step_{s.step:03d}.mp4" for s in guided.steps],
        "dashboard_video": out.name,
    }
    if any(s.extras for s in guided.steps):
        extension_path = out.with_name("extensions.mp4")
        payload["extensions"] = render_extensions(
            guided, extension_path, decode, fps, preview_height, temporal_ratio)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    for obsolete in (out.parent / "baseline_summary.json",
                     out.parent / f"{guided.name}_summary.json"):
        obsolete.unlink(missing_ok=True)
    return out


def render_extensions(trace, path, decode, fps, height, temporal_ratio):
    """Separate synchronized extension panels; legacy dashboard layout stays stable."""
    labels = {
        "motion_support": "Motion support", "historical_disagreement": "Outer-step disagreement",
        "exploration_mask": "Exploration mask", "noise_update": "Applied noise",
        "momentum_update": "Applied momentum", "total_update": "Total post-step change",
        "pnp_disagreement": "Within-step motion disagreement", "confidence": "Preservation confidence",
        "ungated_momentum_update": "Ungated momentum proposal", "refinement_update": "P&P continuation change",
    }
    keys = [k for k in labels if any(k in s.extras for s in trace.steps)]
    update_keys = {k for k in keys if "update" in k}
    scales = {}
    for key in keys:
        maps = []
        for step in trace.steps:
            value = step.extras.get(key)
            if value is not None:
                maps.append((value.square().mean(1).sqrt() if value.ndim == 4 else value).flatten())
        scales[key] = max(float(torch.quantile(torch.cat(maps), .99)), _EPS)
        if key in {"motion_support", "historical_disagreement", "exploration_mask", "confidence"}:
            scales[key] = 1.0
    shared_update_scale = max((scales[k] for k in update_keys), default=1.0)
    for key in update_keys:
        scales[key] = shared_update_scale
    width = max(160, int(height * 1.5))
    size = (width, height)
    steps = np.array([s.step for s in trace.steps], dtype=float)
    series = []
    for key, color in (("momentum_update", (70, 210, 255)), ("noise_update", (255, 180, 60)),
                       ("refinement_update", (150, 255, 100))):
        if key in keys:
            series.append((key.replace("_update", ""), np.array([
                _rms(s.extras[key]) if key in s.extras else 0 for s in trace.steps]), color))

    def frames():
        for step in trace.steps:
            if step.x0 is None:
                continue
            clean = decode(step.x0).detach().float().cpu()
            before = step.extras.get("original_clean")
            before_video = decode(before).detach().float().cpu() if before is not None else clean
            for frame_index in range(len(clean)):
                panels = []
                for video, title in ((before_video, "Original clean prediction"), (clean, "Final clean prediction")):
                    rgb = (video[frame_index].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    panels.append(_caption(cv2.resize(rgb, size), title))
                for key in keys:
                    value = step.extras.get(key)
                    canvas = np.zeros((height, width, 3), np.uint8)
                    if value is not None:
                        if value.shape[0] == step.current.shape[0]:
                            index = _video_to_transition(frame_index, temporal_ratio, value.shape[0])
                        else:
                            index = min((frame_index + temporal_ratio - 1) // temporal_ratio, value.shape[0] - 1)
                        if index is not None:
                            field = value[index]
                            if field.ndim == 2:
                                field = field[None]
                            canvas = _map(field, scales[key], size)
                    panels.append(_caption(canvas, labels[key]))
                while len(panels) % 3:
                    panels.append(np.zeros_like(panels[0]))
                grid = np.concatenate([np.concatenate(panels[j:j+3], axis=1)
                                       for j in range(0, len(panels), 3)], axis=0)
                chart = _chart(series, f"Step {step.step}: intervention RMS", width * 3, 140,
                               steps, step.step, "denoising step", (float(steps.min()), float(steps.max())))
                yield np.concatenate((grid, chart), axis=0)
    encode_frames(frames(), path, fps=fps)
    return {"video": path.name, "scales_p99": scales,
            "interpretation": "Outer-step residual history and within-step P&P disagreement are distinct heuristics, not physics uncertainty."}


class StepPredictionCapture:
    """Recover the CFG-combined denoiser state at the exact input of each step."""

    def __init__(self, backend: Any, guidance_scale: float):
        self.backend = backend
        self.guidance_scale = guidance_scale
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.handle = None
        self.total_calls = 0

    def __enter__(self):
        def hook(_module, args, kwargs, output):
            self.total_calls += 1
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            prediction = output[0] if isinstance(output, (tuple, list)) else output
            self.calls.append((hidden.detach(), prediction.detach()))

        self.handle = self.backend.pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()
        self.calls.clear()

    def consume(self, timestep: torch.Tensor):
        try:
            if not self.calls:
                return None
            latents, output = self.backend.combine_cfg(self.calls, self.guidance_scale)
            return self.backend.denoiser_state(latents, output, timestep)
        finally:
            self.calls.clear()
