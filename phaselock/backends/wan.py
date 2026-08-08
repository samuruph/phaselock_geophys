"""Wan2.1 backend.

Differs from CogVideoX in every way that matters downstream: latents are
``(B, C, T, H, W)``, normalisation is per-channel (``AutoencoderKLWan`` has no
meaningful ``scaling_factor``), and the transformer predicts a flow-matching
velocity rather than ``v`` under a VP schedule.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import CFG_SEQUENTIAL, DenoiserState, LatentSpec, VideoBackend, to_canonical

# Verified against vae/config.json of both Wan2.1-T2V-1.3B-Diffusers and
# Wan2.1-T2V-14B-Diffusers, which carry identical constants.
_WAN_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
_WAN_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def _wan_spec(name: str, height: int = 480, width: int = 832, flow_shift: float = 3.0) -> LatentSpec:
    return LatentSpec(
        name=name,
        layout="BCTHW",
        temporal_ratio=4,
        spatial_ratio=8,
        channels=16,
        patch_size=(1, 2, 2),
        default_num_frames=81,
        default_fps=16,
        default_height=height,
        default_width=width,
        latents_mean=_WAN_LATENTS_MEAN,
        latents_std=_WAN_LATENTS_STD,
        flow_shift=flow_shift,
    )


WAN21_T2V_1_3B = _wan_spec("wan2.1-t2v-1.3b")
WAN21_T2V_14B = _wan_spec("wan2.1-t2v-14b")
WAN21_I2V_14B_480P = _wan_spec("wan2.1-i2v-14b-480p")
# The 720P I2V checkpoint is trained at a larger flow shift.
WAN21_I2V_14B_720P = _wan_spec("wan2.1-i2v-14b-720p", height=720, width=1280, flow_shift=5.0)


class WanBackend(VideoBackend):
    """Wan2.1, in either text-to-video or image-to-video mode.

    Only the T2V-1.3B path is exercised on the current single-GPU box; the 14B specs
    are registered so configs and layout handling are correct, but they are too large
    to validate here.
    """

    def __init__(self, pipe: Any, spec: LatentSpec, device=None, mode: str = "t2v"):
        super().__init__(pipe, spec, device)
        self.mode = mode
        self._validate_vae_constants()

    def _validate_vae_constants(self) -> None:
        """Fail loudly if the checkpoint's normalisation differs from the spec."""
        config = getattr(self.pipe, "vae", None)
        if config is None:
            return
        config = config.config
        for field, expected in (("latents_mean", self.spec.latents_mean), ("latents_std", self.spec.latents_std)):
            actual = getattr(config, field, None)
            if actual is None:
                continue
            if len(actual) != len(expected) or any(
                abs(a - b) > 1e-6 for a, b in zip(actual, expected)
            ):
                raise ValueError(
                    f"{self.spec.name}: checkpoint {field} does not match the registered spec. "
                    f"Update phaselock/backends/wan.py rather than silently mis-normalising."
                )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        spec: LatentSpec = WAN21_T2V_1_3B,
        mode: str = "t2v",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_offload: bool = True,
        **kwargs: Any,
    ) -> "WanBackend":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        if mode not in ("t2v", "i2v"):
            raise ValueError(f"mode must be 't2v' or 'i2v', got {mode!r}")
        pipeline_class = WanPipeline if mode == "t2v" else WanImageToVideoPipeline

        # The Wan VAE is kept in fp32: it is small, and its per-channel normalisation
        # is numerically fragile in bf16.
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = pipeline_class.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype, **kwargs)
        if spec.flow_shift is not None and hasattr(pipe.scheduler.config, "flow_shift"):
            pipe.scheduler.config.flow_shift = spec.flow_shift
        if enable_offload:
            pipe.enable_model_cpu_offload()
        else:
            # Without offload diffusers leaves every component on the CPU; the caller
            # asked for speed, not for a pipeline that raises on the first forward.
            pipe.to(cls._resolve_device())
        # Encoding a full clip at native resolution in one piece is the peak memory point
        # of the whole pipeline, above the transformer itself.
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        return cls(pipe, spec, mode=mode)

    # -- model structure ---------------------------------------------------

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self.pipe.transformer.blocks

    @property
    def cfg_style(self) -> str:
        # Two calls per step: the conditional prediction first, then the unconditional.
        return CFG_SEQUENTIAL

    @staticmethod
    def block_hidden_states(block_output: Any) -> torch.Tensor:
        # WanTransformerBlock returns the hidden states directly; text tokens are
        # never concatenated into them.
        return block_output

    # -- VAE ---------------------------------------------------------------

    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        posterior = vae.encode(self._to_vae(video, vae)).latent_dist
        return posterior.mode()  # already (B, C, T, h, w) == BCTHW

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        return vae.decode(self._to_vae(latents, vae)).sample

    # -- denoiser ----------------------------------------------------------

    def denoiser_state(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> DenoiserState:
        """Invert the flow-matching parameterisation.

        Wan's scheduler runs ``sigma`` from 1 (noise) to 0 (data) over the rectified
        path ``z = (1-sigma)*x0 + sigma*eps``, and the network predicts
        ``v = eps - x0``. Hence::

            x0  = z - sigma*v
            eps = z + (1-sigma)*v
        """
        z = to_canonical(latents, self.spec).float()
        v = to_canonical(model_output, self.spec).float()

        t = float(timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep)
        sigma = t / self.num_train_timesteps

        x0 = z - sigma * v
        eps = z + (1.0 - sigma) * v
        return DenoiserState(latents=z, x0=x0, eps=eps, tau=1.0 - sigma)

    def renoise(
        self, x0: torch.Tensor, eps: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        """``z = (1 - sigma) x0 + sigma eps`` -- the rectified flow path."""
        t = float(timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep)
        sigma = t / self.num_train_timesteps
        return (1.0 - sigma) * x0 + sigma * eps

    def prepare_conditioning(
        self,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        *,
        image_embeds: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=False,
            device=self.device,
        )
        return {
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_image": image_embeds,
            "condition_latents": condition_latents,
        }

    @torch.no_grad()
    def transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> torch.Tensor:
        transformer = self.pipe.transformer
        model_input = latents.to(transformer.dtype)

        condition_latents = conditioning.get("condition_latents")
        if condition_latents is not None:
            # Wan I2V conditions by concatenating a mask and the condition latents on
            # the channel axis, which is dim 1 under BCTHW.
            model_input = torch.cat([model_input, condition_latents.to(transformer.dtype)], dim=1)

        if timestep.ndim == 0:
            timestep = timestep.expand(model_input.shape[0])

        image_embeds = conditioning.get("encoder_hidden_states_image")
        return transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=conditioning["encoder_hidden_states"].to(transformer.dtype),
            encoder_hidden_states_image=(
                image_embeds.to(transformer.dtype) if image_embeds is not None else None
            ),
            return_dict=False,
        )[0].float()
