"""CogVideoX backend.

Latents are ``(B, T, C, H, W)``, normalised by a scalar ``scaling_factor``, and the
transformer predicts ``v`` under a VP (DDIM) schedule.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import (
    CFG_BATCHED,
    DenoiserState,
    LatentSpec,
    VideoBackend,
    from_canonical,
    to_canonical,
)

# CogVideoX-5B. patch_size is spatial-only in the 1.0 models, expressed here as the
# equivalent 3-D patch so that per-latent-frame token pooling is uniform across backends.
COGVIDEOX_5B = LatentSpec(
    name="cogvideox-5b",
    layout="BTCHW",
    temporal_ratio=4,
    spatial_ratio=8,
    channels=16,
    patch_size=(1, 2, 2),
    default_num_frames=49,
    default_fps=8,
    default_height=480,
    default_width=720,
    scaling_factor=0.7,
)


class CogVideoXBackend(VideoBackend):
    """CogVideoX-5B, in either text-to-video or image-to-video mode."""

    def __init__(self, pipe: Any, spec: LatentSpec = COGVIDEOX_5B, device=None, mode: str = "t2v"):
        super().__init__(pipe, spec, device)
        self.mode = mode

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        spec: LatentSpec = COGVIDEOX_5B,
        mode: str = "t2v",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_offload: bool = True,
        **kwargs: Any,
    ) -> "CogVideoXBackend":
        from diffusers import CogVideoXImageToVideoPipeline, CogVideoXPipeline

        if mode not in ("t2v", "i2v"):
            raise ValueError(f"mode must be 't2v' or 'i2v', got {mode!r}")
        pipeline_class = CogVideoXPipeline if mode == "t2v" else CogVideoXImageToVideoPipeline
        pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        if enable_offload:
            pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        return cls(pipe, spec, mode=mode)

    # -- model structure ---------------------------------------------------

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self.pipe.transformer.transformer_blocks

    @property
    def cfg_style(self) -> str:
        # One call per step on cat([latents]*2), with prompt_embeds ordered
        # cat([negative, positive]), so chunk(2) yields (uncond, cond).
        return CFG_BATCHED

    @staticmethod
    def block_hidden_states(block_output: Any) -> torch.Tensor:
        # CogVideoXBlock returns (hidden_states, encoder_hidden_states); the video
        # tokens are already separated from the text tokens by the parent module.
        return block_output[0]

    # -- VAE ---------------------------------------------------------------

    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        posterior = vae.encode(self._to_vae(video, vae)).latent_dist
        latents = posterior.mode()
        return latents.permute(0, 2, 1, 3, 4)  # (B, C, T, h, w) -> BTCHW

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        latents = latents.permute(0, 2, 1, 3, 4)  # BTCHW -> (B, C, T, h, w)
        return vae.decode(self._to_vae(latents, vae)).sample

    # -- denoiser ----------------------------------------------------------

    def denoiser_state(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> DenoiserState:
        """Invert the v-prediction parameterisation.

        With ``z = sqrt(a)*x0 + sqrt(1-a)*eps`` and ``v = sqrt(a)*eps - sqrt(1-a)*x0``::

            x0  = sqrt(a)*z - sqrt(1-a)*v
            eps = sqrt(a)*v + sqrt(1-a)*z
        """
        z = to_canonical(latents, self.spec).float()
        v = to_canonical(model_output, self.spec).float()

        t = int(timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep)
        alpha_bar = self.pipe.scheduler.alphas_cumprod.to(z.device).float()[t]
        sqrt_a = alpha_bar.sqrt()
        sqrt_1ma = (1.0 - alpha_bar).sqrt()

        x0 = sqrt_a * z - sqrt_1ma * v
        eps = sqrt_a * v + sqrt_1ma * z
        return DenoiserState(latents=z, x0=x0, eps=eps, tau=1.0 - t / self.num_train_timesteps)

    def renoise(
        self, x0: torch.Tensor, eps: torch.Tensor, timestep: torch.Tensor | float
    ) -> torch.Tensor:
        """``z_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps`` -- the VP forward process."""
        t = int(timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep)
        alpha_bar = self.pipe.scheduler.alphas_cumprod.to(x0.device).float()[t]
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * eps

    def prepare_conditioning(
        self,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        *,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image_latents: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from .base import num_latent_frames

        num_frames = num_frames or self.spec.default_num_frames
        height = height or self.spec.default_height
        width = width or self.spec.default_width

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=False,
            device=self.device,
        )

        image_rotary_emb = None
        if self.pipe.transformer.config.use_rotary_positional_embeddings:
            image_rotary_emb = self.pipe._prepare_rotary_positional_embeddings(
                height, width, num_latent_frames(num_frames, self.spec), self.device
            )

        ofs_emb = None
        if getattr(self.pipe.transformer.config, "ofs_embed_dim", None) is not None:
            ofs_emb = torch.full((1,), 2.0, device=self.device)

        return {
            "encoder_hidden_states": prompt_embeds,
            "image_rotary_emb": image_rotary_emb,
            "ofs": ofs_emb,
            "image_latents": image_latents,
        }

    def transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> torch.Tensor:
        transformer = self.pipe.transformer
        model_input = latents.to(transformer.dtype)

        image_latents = conditioning.get("image_latents")
        if image_latents is not None:
            # I2V conditions by concatenating the image latents on the channel axis,
            # which is dim 2 under BTCHW.
            model_input = torch.cat([model_input, image_latents.to(transformer.dtype)], dim=2)

        if timestep.ndim == 0:
            timestep = timestep.expand(model_input.shape[0])

        return transformer(
            hidden_states=model_input,
            encoder_hidden_states=conditioning["encoder_hidden_states"].to(transformer.dtype),
            timestep=timestep,
            ofs=conditioning.get("ofs"),
            image_rotary_emb=conditioning.get("image_rotary_emb"),
            return_dict=False,
        )[0].float()
