import torch
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)


class VanillaStableDiffusion(torch.nn.Module):
    def __init__(self, pretrained, auth_token):
        super().__init__()


        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            pretrained,
            subfolder="vae",
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=auth_token,
        )

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained,
            subfolder="unet",
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=auth_token,
        )

    def forward(self, z, c, uc, w=7.5, width=512, height=512, num_timesteps=64):

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=num_timesteps,
        )

        cond_uncond = torch.cat([uc, c])

        scheduler.set_timesteps(num_timesteps)
        latents = latents * scheduler.init_noise_sigma

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=cond_uncond
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + w * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
