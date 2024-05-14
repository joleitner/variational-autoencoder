import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import typer
from typing import Annotated
from rich import print
from rich.progress import track
from models.conv_vae import ConvVAE
from utils import load_dataset, save_model_config, save_checkpoint


def _validate_hidden_dims(hidden_dims: str):
    try:
        valid = [int(item) for item in hidden_dims.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide hidden dimensions as comma-separated integers")
    return hidden_dims


def _validate_resize(resize: str):
    if resize is None:
        return None
    try:
        valid = [int(item) for item in resize.split(",")]
        if len(valid) != 2:
            raise ValueError
    except ValueError:
        raise typer.BadParameter("Please provide resize dimensions as comma-separated integers")
    return resize


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def train(
    data_path: Annotated[
        str,
        typer.Option(
            "--data-path",
            "-d",
            help="Path to the data",
            rich_help_panel="Training parameters",
            prompt="Path to the dataset to train on",
            show_default=False,
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size",
            rich_help_panel="Training parameters",
            prompt="Batch size for training",
        ),
    ] = 32,
    resize: Annotated[
        str,
        typer.Option(
            help="Resize of original image",
            rich_help_panel="Training parameters",
            # prompt="Resize input image (width, height)",
            callback=_validate_resize,
        ),
    ] = "48,48",
    hidden_dims: Annotated[
        str,
        typer.Option(
            help="Dimensions of hidden layer(s) For multiple layers, separate by comma (e.g. 512,256)",
            rich_help_panel="Model Parameters",
            prompt="Hidden Dimensions (comma separated)",
            callback=_validate_hidden_dims,
        ),
    ] = "32,64,64",
    latent_dim: Annotated[
        int,
        typer.Option(
            help="Dimension of lantent space (z)",
            rich_help_panel="Model Parameters",
            prompt="Latent Dimensions",
        ),
    ] = 10,
    epochs: Annotated[
        int,
        typer.Option(
            help="Epochs to train", rich_help_panel="Training parameters", prompt="Epochs"
        ),
    ] = 10,
    save: Annotated[
        str,
        typer.Option(
            help="Path to save the model",
            rich_help_panel="Model Parameters",
            prompt="Where would you like to save the model?",
        ),
    ] = "models/my_model",
):
    """
    Define and train a Convolutional Variational Autoencoder on a given dataset.


    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # resize string to tuple
    if resize:
        resize = tuple([int(item) for item in resize.split(",")])

    data_loader = load_dataset(data_path, batch_size, resize, grayscale=False)
    images, _ = next(iter(data_loader))
    # Convert the hidden_dims string to a list of integers
    hidden_dims = [int(item) for item in hidden_dims.split(",")]

    model = ConvVAE(hidden_dims=hidden_dims, latent_dim=latent_dim, image_shape=images[0].shape).to(
        device
    )

    print("Your created Model:")
    print(model)
    print(f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    for epoch in track(
        range(epochs), description="Training process"
    ):  # track is showing a progress bar

        overall_loss = 0
        for x, _ in data_loader:

            x = x.to(device)

            optimizer.zero_grad()
            x_hat, mean, log_var, _ = model(x)
            loss = model.loss_function(x_hat, x, mean, log_var)
            # overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        # loss = overall_loss / len(data_loader.dataset)
        print("\tEpoch", epoch + 1, "\tAverage Loss: {:.3f}".format(loss))

    # Save the model
    config = {
        # "initial_dim": initial_dim,
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "data_shape": images[0].shape,
        "grayscale": True,
    }
    if resize:
        config["resize"] = resize

    save_model_config(config, path=save)
    save_checkpoint(model, optimizer, epochs, loss, path=save)
    print(f"Model saved to [bold green]{save}[/bold green] :floppy_disk:")


if __name__ == "__main__":
    app()
