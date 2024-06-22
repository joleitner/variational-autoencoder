import typer
from typing import Annotated
from rich import print

import src.utils as utils
from src.train_model import train_model
from src.load_model import load_model


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def resume_training(
    model_path: Annotated[
        str,
        typer.Argument(help="To continue training, provide path to the model"),
    ],
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
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Epochs to resume training",
            rich_help_panel="Training parameters",
            prompt="Epochs",
        ),
    ] = 10,
):
    """
    Resume training of a previously trained model.
    """

    config = utils.load_model_config(model_path)
    data_loader = utils.load_dataset(
        data_path, config["batch_size"], config["resize"], config["grayscale"]
    )

    model, device = load_model(model_path)
    checkpoint = utils.load_checkpoint(model_path)

    print(f"Training continuing on [bold green]{str(device).upper()}[/bold green]")

    train_model(
        model=model,
        data_loader=data_loader,
        epochs=epochs,
        checkpoint=checkpoint,
        save_path=model_path,
    )

    print(f"New checkpoint saved to [bold green]{model_path}[/bold green] :floppy_disk:")
