import matplotlib.pyplot as plt
import os
import typer
from typing import Annotated
from rich import print

import src.utils as utils
import src.cli.utils as cli_utils
from src.load_model import load_model


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def generate(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the trained model"),
    ],
    latent_vector: Annotated[
        str,
        typer.Option(
            "--latent-vector",
            "-v",
            help="Number of latent variables to generate from (e.g. 1.0,2.3 for 2 dimensional latent space)",
            prompt="Latent vector (comma separated)",
            callback=cli_utils.validate_latent_vector,
        ),
    ],
    save: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path where to save the generated image",
            prompt="Path to save the generated image",
        ),
    ] = "images/image.jpg",
    plot: Annotated[bool, typer.Option(help="Plot the generated image")] = True,
):
    """
    Generate an image from a given latent vector using a trained VAE model.
    """

    # convert the latent vector to list of floats
    latent_vector = cli_utils.convert_latent_vector(latent_vector)
    model, device = load_model(model_path)
    output = model.generate(latent_vector, device)

    image = utils.convert_tensor_to_image(output)

    # Save the generated image
    save_dir = os.path.dirname(save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image.save(save)

    if plot:
        plt.imshow(image, cmap="gray" if image.mode == "L" else None)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    app()
