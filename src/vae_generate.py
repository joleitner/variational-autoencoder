import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import typer
from typing import Annotated
from rich import print
from utils import load_model_config, load_checkpoint
from model import VAE


def _validate_latent_vector(latent_vector: str):
    try:
        valid = [float(item) for item in latent_vector.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide latent vector as comma-separated floats")
    return latent_vector


def _reshape_image(output, data_shape):

    if len(data_shape) != 3:
        raise ValueError("data_shape must be a tuple of length 3 (channels, width, height).")

    # Reshape the data based on the number of channels
    if data_shape[0] == 1:  # Grayscale image
        image_array = output.detach().cpu().reshape(data_shape[1], data_shape[2]).numpy()
        mode = "L"
    elif data_shape[0] == 3:  # RGB image
        image_array = (
            output.detach().cpu().reshape(data_shape[1], data_shape[2], data_shape[0]).numpy()
        )
        mode = "RGB"
    else:  # Invalid number of channels
        raise ValueError(
            "Unsupported number of channels in data_shape. Only grayscale (1) or RGB (3) images supported."
        )

    # Assuming model output is not normalized and could have any range
    # We normalize to 0-1 range and then scale to 0-255
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode=mode)
    return image


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def generate(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the trained model"),
    ],
    latent_vector: Annotated[
        str,
        typer.Option(
            "--latent-vector",
            "-v",
            help="Number of latent variables to generate from (e.g. 1.0,2.3 for 2 dimensional latent space)",
            prompt="Latent vector (comma separated)",
            callback=_validate_latent_vector,
        ),
    ],
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path where to save the generated image",
            prompt="Path to save the generated image",
        ),
    ] = "images/image.jpg",
    plot: Annotated[bool, typer.Option(help="Plot the generated image")] = True,
):
    """
    Generate an image from a given latent vector using a trained VAE model.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_model_config(model_path)
    checkpoint = load_checkpoint(model_path)
    model = VAE(
        initial_dim=config["initial_dim"],
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # convert the latent vector to list of floats
    latent_vector = [float(item) for item in latent_vector.split(",")]
    # convert the latent vector to tensor
    latent_vector = torch.tensor(latent_vector).to(device)

    output = model.decoder(latent_vector)

    image = _reshape_image(output, config["data_shape"])

    # Save the generated image
    save_dir = os.path.dirname(save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image.save(save)

    if plot:
        plt.imshow(image, cmap="gray" if config["data_shape"][0] == 1 else None)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    app()
