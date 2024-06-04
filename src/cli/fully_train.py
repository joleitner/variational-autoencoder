import typer
from typing import Annotated
from rich import print
from rich.progress import track

import src.utils as utils
import src.cli.utils as cli_utils
from src.create_model import create_vae_model
from src.train_model import train_model


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def train_fully_connected(
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
            help="Resize of original image [width, height]",
            rich_help_panel="Training parameters",
            callback=cli_utils.validate_resize,
        ),
    ] = None,
    hidden_dims: Annotated[
        str,
        typer.Option(
            help="Dimensions of hidden layer(s) For multiple layers, separate by comma (e.g. 512,256)",
            rich_help_panel="Model Parameters",
            prompt="Hidden Dimensions (comma separated)",
            callback=cli_utils.validate_hidden_dims,
        ),
    ] = "512",
    latent_dim: Annotated[
        int,
        typer.Option(
            help="Dimension of lantent space (z)",
            rich_help_panel="Model Parameters",
            prompt="Latent Dimensions",
        ),
    ] = 3,
    epochs: Annotated[
        int,
        typer.Option(
            help="Epochs to train", rich_help_panel="Training parameters", prompt="Epochs"
        ),
    ] = 30,
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
    Define and train a fully-connected Variational Autoencoder model.
    """

    # convert input strings to appropriate types
    hidden_dims = cli_utils.convert_hidden_dims(hidden_dims)
    if resize:
        resize = cli_utils.convert_resize(resize)

    data_loader = utils.load_dataset(data_path, batch_size, resize, grayscale=True)
    images, _ = next(iter(data_loader))
    image_shape = images[0].shape

    model, device = create_vae_model(
        model_type="fully",
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        image_shape=image_shape,
    )

    print("Your created Model:")
    print(model)
    print(f"Training starting on [bold green]{str(device).upper()}[/bold green]")

    model, optimizer, loss = train_model(model=model, data_loader=data_loader, epochs=epochs)

    # Save the model
    config = {
        "hidden_dims": hidden_dims,
        "latent_dim": latent_dim,
        "image_shape": image_shape,
        "grayscale": True,
        "model_type": "fully",
        "epochs": epochs,
    }
    if resize:
        config["resize"] = resize

    utils.save_model_config(config, path=save)
    utils.save_checkpoint(model, optimizer, epochs, loss, path=save)
    print(f"Model saved to [bold green]{save}[/bold green] :floppy_disk:")


if __name__ == "__main__":
    app()
