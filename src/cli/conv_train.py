import typer
from typing import Annotated
from rich import print

import src.utils as utils
import src.cli.utils as cli_utils
from src.create_model import create_vae_model
from src.train_model import train_model


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def train_convolutional(
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
            "--batch-size",
            "-b",
            help="Batch size",
            rich_help_panel="Training parameters",
            prompt="Batch size for training",
        ),
    ] = 32,
    resize: Annotated[
        str,
        typer.Option(
            "--resize",
            "-r",
            help="Resize of original image [width, height]",
            rich_help_panel="Training parameters",
            # prompt="Resize image (e.g. 64,64)",
            callback=cli_utils.validate_resize,
        ),
    ] = None,
    grayscale: Annotated[
        bool,
        typer.Option(
            "--gray/--rgb",
            help="Convert image to grayscale",
            rich_help_panel="Training parameters",
        ),
    ] = False,
    hidden_dims: Annotated[
        str,
        typer.Option(
            "--hidden-dims",
            "-hdims",
            help="Dimensions of hidden layer(s) For multiple layers, separate by comma (e.g. 32,64,128)",
            rich_help_panel="Model Parameters",
            prompt="Hidden Dimensions (comma separated)",
            callback=cli_utils.validate_hidden_dims,
        ),
    ] = "32,64,64",
    latent_dim: Annotated[
        int,
        typer.Option(
            "--latent-dim",
            "-ldim",
            help="Dimension of lantent space (z)",
            rich_help_panel="Model Parameters",
            prompt="Latent Dimensions",
        ),
    ] = 3,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Epochs to train",
            rich_help_panel="Training parameters",
            prompt="Epochs",
        ),
    ] = 30,
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path to save the model",
            rich_help_panel="Model Parameters",
            prompt="Where would you like to save the model?",
        ),
    ] = "models/my_conv_model",
):
    """
    Define and train a Convolutional Variational Autoencoder model.
    """

    # convert input strings to appropriate types
    hidden_dims = cli_utils.convert_hidden_dims(hidden_dims)
    if resize:
        resize = cli_utils.convert_resize(resize)

    data_loader = utils.load_dataset(data_path, batch_size, resize, grayscale)
    images, _ = next(iter(data_loader))
    image_shape = images[0].shape

    model, device = create_vae_model(
        model_type="conv",
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        image_shape=image_shape,
    )

    print("Your created Model:")
    print(model)
    print(f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    # Save model configuration
    config = {
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "image_shape": image_shape,
        "resize": resize if resize else image_shape[1:],
        "grayscale": grayscale,
        "model_type": "conv",
        "epochs": epochs,
        "batch_size": batch_size,
        "loss_history": [],
    }

    utils.save_model_config(config, path=save)

    train_model(model=model, data_loader=data_loader, epochs=epochs, save_path=save)

    print(f"Model successfully trained and saved to [bold green]{save}[/bold green] :floppy_disk:")


if __name__ == "__main__":
    app()
