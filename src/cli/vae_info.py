import torch
import pandas as pd
import os
import typer
from typing import Annotated
from rich import print
from rich.table import Table
import json


from src.load_model import load_model
import src.utils as utils


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def info(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the model"),
    ],
    detailed: Annotated[
        bool,
        typer.Option(
            "--detailed",
            "-d",
            help="Show detailed model information",
        ),
    ] = False,
):
    """
    Prints a detailed information about the specified VAE model.
    """

    model, device = load_model(model_path)
    config = utils.load_model_config(model_path)

    table = Table("Attribute", "Value", title="Model Information")
    table.add_row("Model Path", f"[bold green]{model_path}[/bold green]")
    model_type = (
        "Convolutional Layers" if config["model_type"] == "conv" else "Fully Connected Layers"
    )
    table.add_row("Model Type", f"[bold yellow]{model_type}[/bold yellow]")
    latent_dim = config["latent_dim"]
    table.add_row("Latent Dimensions", f"[bold blue]{latent_dim}[/bold blue]")
    hidden_dims = config["hidden_dims"]
    table.add_row("Hidden Dimensions", f"[bold blue]{hidden_dims}[/bold blue]")
    table.add_row("Epochs trained", f"[bold blue]{config['epochs']}[/bold blue]")
    last_loss = round(config["loss_history"][-1], 3) if config["loss_history"] else "N/A"
    table.add_row("Loss", f"[bold red]{last_loss}[/bold red]")

    total_params = sum(param.numel() for param in model.parameters())
    table.add_row("Total of parameters", f"[bold blue]{total_params}[/bold blue]")
    checkpoint_path = os.path.join(model_path, "checkpoint.tar")
    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    table.add_row("Checkpoint size:", f"{file_size_mb:.2f} MB :floppy_disk:")
    print(table)

    if detailed:
        print("\nModel Architecture: :house: \n")
        print(model)


if __name__ == "__main__":
    app()
