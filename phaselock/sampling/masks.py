"""Local motion and disagreement maps. These are not physics uncertainty scores."""
import torch
import torch.nn.functional as F


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    scale = torch.quantile(x.flatten(), 0.9).clamp_min(1e-8)
    return (x / scale).clamp(0, 1)


def transition_to_frames(x: torch.Tensor) -> torch.Tensor:
    """(T-1,H,W) -> (T,H,W), with averaged interior transitions."""
    return torch.cat((x[:1], (x[:-1] + x[1:]) * 0.5, x[-1:]), dim=0)


def support(x: torch.Tensor) -> torch.Tensor:
    x = transition_to_frames(normalize_map(x))[:, None]
    x = F.max_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[:, 0]


def exploration_mask(x0, variance, mode="motion_disagreement", floor=0.05):
    motion = support((x0[1:].float() - x0[:-1].float()).square().mean(1).sqrt())
    disagreement = support(variance.float().clamp_min(0).mean(1).sqrt())
    selected = {"uniform": torch.ones_like(motion), "motion": motion,
                "disagreement": disagreement,
                "motion_disagreement": motion * disagreement}[mode]
    mask = floor + (1 - floor) * selected
    mask[0] = 0
    return mask, motion, disagreement


def motion_confidence(previous, current):
    before = previous[1:].float() - previous[:-1].float()
    after = current[1:].float() - current[:-1].float()
    disagreement = (after - before).abs().mean(1)
    scale = (before.square().mean().sqrt() + after.square().mean().sqrt()) * 0.5
    normalized = disagreement / scale.clamp_min(1e-8)
    return disagreement, 1 / (1 + normalized)
