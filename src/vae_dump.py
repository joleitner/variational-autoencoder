import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch
import pandas as pd
from PIL import Image
import os
import typer
from typing import Annotated
from rich import print
from utils import load_model_config, load_checkpoint
from model import VAE


device = "cuda" if torch.cuda.is_available() else "cpu"


def _input_file_to_tensor(input: str, config: dict):
    transform = []
    if config.get("resize", None):
        transform.append(transforms.Resize(config["resize"]))
    if config.get("grayscale", False):
        transform.append(transforms.Grayscale())
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    # load image
    image = Image.open(input)
    # transform to tensor
    image = transform(image).to(device)
    return image.view(-1, config["initial_dim"]).to(device)


def _validate_input(input: str):

    if os.path.isfile(input):
        if not input.lower().endswith((".png", ".jpg", ".jpeg")):
            raise typer.BadParameter("Please provide a valid image file")

    if not os.path.exists(input):
        raise typer.BadParameter("Please provide a valid path to an image or directory")

    return input


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def dump(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the model"),
    ],
    input: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Path to the input image or directory",
            callback=_validate_input,
        ),
    ],
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path to save the latent dump",
        ),
    ] = "dumps/latent.csv",
):
    """
    Dump the latent vector(s) of a trained VAE model.
    """

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

    if os.path.isfile(input):
        input = _input_file_to_tensor(input, config)

        mean, logvar = model.encoder(input)
        std = torch.sqrt(torch.exp(logvar))
        print("Mean: ", mean.cpu().detach().numpy())
        print("Standard deviation", std.cpu().detach().numpy())
    else:
        # TODO: Implement directory processing
        # dataloader and save to dataframe
        pass


if __name__ == "__main__":
    app()
