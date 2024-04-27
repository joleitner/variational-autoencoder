import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import typer
from typing import Annotated
from rich import print
from model import VAE


def _validate_latent_vector(latent_vector: str):
    try:
        valid = [float(item) for item in latent_vector.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide latent vector as comma-separated floats")
    return latent_vector


def generate(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the model"),
    ],
    latent_vector: Annotated[
        str,
        typer.Option(
            "--latent-vector",
            "-v",
            help="Number of latent variables to generate from (e.g. 1.0,2.3 for 2 dimensional latent space)",
            prompt="Lantent vector (comma separated)",
            callback=_validate_latent_vector,
        ),
    ],
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path to save the generated image",
        ),
    ] = "images/image.jpg",
    plot: Annotated[bool, typer.Option("--plot", help="Plot the generated image")] = False,
):
    """
    Train a Variational Autoencoder
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model and generate the image
    model: VAE = torch.load(model_path)

    # convert the latent vector to list of floats
    latent_vector = [float(item) for item in latent_vector.split(",")]
    # convert the latent vector to tensor
    latent_vector = torch.tensor(latent_vector).to(device)

    x_hat = model.decoder(latent_vector)
    img_array = x_hat.detach().cpu().reshape(28, 28)  # reshape vector to 2d array
    image = Image.fromarray(img_array.numpy() * 255).convert("L")

    # Save the generated image
    save_dir = os.path.dirname(save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image.save(save)

    if plot:
        plt.show()


if __name__ == "__main__":
    typer.run(generate)
