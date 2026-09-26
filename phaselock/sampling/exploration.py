"""Bounded post-step stochastic exploration with an independent RNG stream."""
import torch
import torch.nn.functional as F

from .masks import exploration_mask


def rms(x):
    return x.float().square().mean().sqrt()


def smooth_noise(noise):
    # Canonical T,C,H,W -> convolution layout N,C,T,H,W.
    x = noise.permute(1, 0, 2, 3)[None]
    for axis in (0, 1, 2, 1, 2):
        shape = [1, 1, 1]
        shape[axis] = 3
        kernel = x.new_tensor([0.25, 0.5, 0.25]).reshape(1, 1, *shape)
        kernel = kernel.expand(x.shape[1], 1, *shape)
        padding = [0] * 6
        padding[2 * (2 - axis):2 * (2 - axis) + 2] = [1, 1]
        x = F.conv3d(F.pad(x, tuple(padding), mode="replicate"), kernel, groups=x.shape[1])
    return x[0].permute(1, 0, 2, 3)


class StochasticExploration:
    def __init__(self, config, start, end):
        self.config, self.start, self.end = config, start, end
        self.generator = None

    def apply(self, latents, x0, variance, step, noise_coefficient):
        cfg = self.config
        if not cfg.enabled or cfg.noise_ratio == 0 or not self.start <= step < self.end or noise_coefficient == 0:
            return latents, {}
        if latents.shape[0] < 2:
            return latents, {}
        if self.generator is None:
            self.generator = torch.Generator(device=latents.device).manual_seed(cfg.seed)
        mask, motion, disagreement = exploration_mask(x0, variance, cfg.mask_mode, cfg.floor)
        noise = torch.randn(latents.shape, generator=self.generator, device=latents.device, dtype=torch.float32)
        structured = smooth_noise(noise)
        structured[1:] -= structured[1:].mean(0, keepdim=True)
        structured[0] = 0
        structured /= rms(structured[1:]).clamp_min(1e-8)
        amplitude = cfg.noise_ratio * (1 - (step - self.start) / (self.end - self.start))
        latent_rms = rms(latents[1:])
        proposed = amplitude * noise_coefficient * latent_rms * mask[:, None] * structured
        if not cfg.structured:
            # Same draw and exact masked proposal RMS as the structured control.
            unstructured = noise * mask[:, None]
            proposed = unstructured * (rms(proposed[1:]) / rms(unstructured[1:]).clamp_min(1e-8))
        before_cap = rms(proposed[1:])
        update = proposed * (0.05 * latent_rms / before_cap.clamp_min(1e-8)).clamp(max=1)
        guided = latents + update.to(latents.dtype)
        actual = guided.float() - latents.float()
        return guided, {
            "motion_support": motion, "historical_disagreement": disagreement,
            "exploration_mask": mask, "noise_update": actual,
            "noise_proposed_rms": float(before_cap), "noise_capped_rms": float(rms(update[1:])),
            "noise_actual_rms": float(rms(actual[1:])),
            "mask_mean": float(mask[1:].mean()), "mask_active_fraction": float((mask[1:] > cfg.floor).float().mean()),
            "noise_anchor_norm": float(actual[0].norm()),
        }
