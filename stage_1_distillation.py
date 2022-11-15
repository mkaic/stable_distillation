import torch
import argparse

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from vanilla_sd import VanillaStableDiffusion#, FusedGuidanceStableDiffusion


def main(args):
    with open("token.txt", "r") as f:
        auth_token = f.read()
    teacher = VanillaStableDiffusion(pretrained="runwayml/stable-diffusion-v1-5", auth_token=auth_token).eval()
    teacher.unet.save_config("teacher_config.json")

    # student = FusedGuidanceStableDiffusion()

    # with torch.no_grad():
    #     # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    #     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    #     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    #     teacher = teacher.eval()
    #     student = student.train()
    #     optimizer = torch.optim.Adam(student.parameters(), lr=1e-5)
    #     loss = torch.nn.MSELoss()

    #     latents = torch.randn(
    #         (args.batch_size, teacher.unet.in_channels, height // 8, width // 8),
    #         generator=generator,
    #     )
    # # with torch.no_grad():

    # #     # scale and decode the image latents with vae
    # #     latents = 1 / 0.18215 * latents
    # #     image = vae.decode(latents).sample

    # # image = (image / 2 + 0.5).clamp(0, 1)
    # # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    # # images = (image * 255).round().astype("uint8")
    # # pil_images = [Image.fromarray(image) for image in images]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main(args)
