import torch.distributions
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import typer
from typing import Annotated
from rich import print
from model import VAE
from utils import load_dataset


def loss_function(x, x_hat, kld):
    # also MSE possible: ((x - x_hat)**2).sum()
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # kl divergence calculated in the encoder
    return reproduction_loss + kld


def train(model, optimizer, data_loader, epochs, device, initial_dim):

    for epoch in range(epochs):

        overall_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):

            x = x.view(-1, initial_dim).to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_function(x, x_hat, kld=model.encoder.kld)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch",
            epoch + 1,
            "\tAverage Loss: {:.3f}".format(overall_loss / len(data_loader.dataset)),
        )

    return model


def _validate_hidden_dims(hidden_dims: str):
    try:
        valid = [int(item) for item in hidden_dims.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide hidden dimensions as comma-separated integers")
    return hidden_dims


def main(
    data_path: Annotated[
        str,
        typer.Option(
            "--data-path",
            "-d",
            help="Path to the data",
            rich_help_panel="Train data",
            prompt="Path to the dataset to train on",
            show_default=False,
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size", rich_help_panel="Train data", prompt="Batch size for training"
        ),
    ] = 32,
    resize: Annotated[
        tuple[int, int],
        typer.Option(help="Resize of original image", rich_help_panel="Train data"),
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
    ] = "models/model.pth",
):
    """
    Train a Variational Autoencoder


    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    model = train(model, optimizer, data_loader, epochs, device, initial_dim)

    # Save model
    save_dir = os.path.dirname(save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model, save)
    print("Model saved to", save)


if __name__ == "__main__":
    # typer.Typer(pretty_exceptions_show_locals=False)
    typer.run(main)
