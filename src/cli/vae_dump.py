import torch.distributions
from torchvision import transforms
import torch
import pandas as pd
from PIL import Image
import os
import typer
from typing import Annotated
from rich import print

from src.load_model import load_model
import src.utils as utils
import src.cli.utils as cli_utils


def _validate_save(save: str):
    if not save.lower().endswith(".csv"):
        raise typer.BadParameter(
            "Please provide a valid path to a CSV File to save the latent dump"
        )
    return save


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
            prompt="Path to a input image or directory",
            callback=cli_utils.validate_image_input,
        ),
    ],
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path to save the latent dump",
            prompt="Path to save the latent dump",
            callback=_validate_save,
        ),
    ] = "dumps/latent.csv",
):
    """
    Dump the latent vector(s) of a trained VAE model.
    """

    model, device = load_model(model_path)
    config = utils.load_model_config(model_path)

    if os.path.isfile(input):
        image = utils.image_to_tensor(
            input, resize=config.get("resize", None), grayscale=config["grayscale"]
        )
        mean, std = model.dump_latent(image, device)
        print("Mean: ", mean)
        print("Standard deviation", std)

    else:
        dumps = []
        for root, _, files in os.walk(input):
            class_name = os.path.basename(root)
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(root, file)
                    image = utils.image_to_tensor(
                        image_path, resize=config.get("resize", None), grayscale=config["grayscale"]
                    )
                    mean, std = model.dump_latent(image, device)
                    dumps.append([image_path, class_name, mean, std])

        dumps = pd.DataFrame(dumps, columns=["Image", "label", "Mean", "Std"])
        dir_path = os.path.dirname(save)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dumps.to_csv(save, index=False)
        print(f"Latent vectors saved to {save}")


if __name__ == "__main__":
    app()
