import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import typer
from typing_extensions import Annotated
from rich import print
from model import VAE

INITAL_DIM = 784


def load_data(path, batch_size):
    mnist_dataset = torchvision.datasets.MNIST(path,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True)
    return data_loader


def loss_function(x, x_hat, kld):
    # also MSE possible: ((x - x_hat)**2).sum()
    reproduction_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='sum')
    # kl divergence calculated in the encoder
    return reproduction_loss + kld


def train(model, optimizer, data_loader, epochs, device):

    for epoch in range(epochs):

        overall_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):

            x = x.view(-1, INITAL_DIM).to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_function(x, x_hat, kld=model.encoder.kld)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1,
              "\tAverage Loss: {:.3f}".format(overall_loss / len(data_loader.dataset)))

    return model


def main(
        data_path: Annotated[str, typer.Option(
            "--data-path", "-d",
            help="Path to the data",
            rich_help_panel="Train data",
            prompt="Path to the dataset to train on",
            show_default=False
        )],
        batch_size: Annotated[int, typer.Option(
            help="Batch size",
            rich_help_panel="Train data",
            prompt="Batch size for training"
        )] = 128,
        hidden_dim: Annotated[int, typer.Option(
            help="Dimensions of hidden layer", rich_help_panel="Model Parameters")] = 512,
        latent_dim: Annotated[int, typer.Option(
            help="Dimension of lantent space (z)", rich_help_panel="Model Parameters")] = 2,
        epochs: Annotated[int, typer.Option(
            help="Epochs to train", rich_help_panel="Training parameters")] = 10,
        save: Annotated[str, typer.Option(
            help="Path to save the model",
            rich_help_panel="Model Parameters",
            prompt="Where would you like to save the model?",
        )] = 'models/model.pth'
):
    """
    Train a Variational Autoencoder


    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = load_data(data_path, batch_size)

    model = VAE(initial_dim=INITAL_DIM, hidden_dim=hidden_dim,
                latent_dim=latent_dim, device=device).to(device)
    print("Your created Model:")
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(
        f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    model = train(model, optimizer, data_loader, epochs, device)

    # Save model
    save_dir = os.path.dirname(save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model, save)
    print("Model saved to", save)


if __name__ == "__main__":
    typer.run(main)
