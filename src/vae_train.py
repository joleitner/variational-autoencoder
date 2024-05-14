import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import typer
from typing import Annotated
from rich import print
from rich.progress import track
from model import VAE
from utils import load_dataset, save_model_config, save_checkpoint


def loss_function(x, x_hat, kld):
    # also MSE possible: ((x - x_hat)**2).sum()
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # kl divergence calculated in the encoder
    return reproduction_loss + kld


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
    ] = None,
    hidden_dims: Annotated[
        str,
        typer.Option(
            help="Dimensions of hidden layer(s) For multiple layers, separate by comma (e.g. 512,256)",
            rich_help_panel="Model Parameters",
            prompt="Hidden Dimensions (comma separated)",
            callback=_validate_hidden_dims,
        ),
    ] = "512",
    latent_dim: Annotated[
        int,
        typer.Option(
            help="Dimension of lantent space (z)",
            rich_help_panel="Model Parameters",
            prompt="Latent Dimensions",
        ),
    ] = 2,
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
    Define and train a Variational Autoencoder on a given dataset.


    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # resize string to tuple
    if resize:
        resize = tuple([int(item) for item in resize.split(",")])

    data_loader = load_dataset(data_path, batch_size, resize, grayscale=True)
    # Calculate the initial dimensions (input size) for the VAE model
    images, _ = next(iter(data_loader))
    initial_dim = images[0].view(-1).shape[0]
    # Convert the hidden_dims string to a list of integers
    hidden_dims = [int(item) for item in hidden_dims.split(",")]

    model = VAE(
        initial_dim=initial_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, device=device
    ).to(device)

    print("Your created Model:")
    print(model)
    print(f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    for epoch in track(
        range(epochs), description="Training process"
    ):  # track is showing a progress bar

        overall_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):

            x = x.view(-1, initial_dim).to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_function(x, x_hat, kld=model.encoder.kld)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        loss = overall_loss / len(data_loader.dataset)
        print("\tEpoch", epoch + 1, "\tAverage Loss: {:.3f}".format(loss))

    # Save the model
    config = {
        "initial_dim": initial_dim,
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
