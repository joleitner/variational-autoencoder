import os
import typer
from typing import Annotated

import src.utils as utils
import src.cli.utils as cli_utils
from src.load_model import load_model


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def sample(
    model_path: Annotated[
        str,
        typer.Argument(help="Path to the trained model"),
    ],
    quantity: Annotated[
        int,
        typer.Option(
            "--quantity",
            "-q",
            help="Number of samples to generate",
        ),
    ] = 5,
    save_dir: Annotated[
        str,
        typer.Option(
            "--save",
            "-s",
            help="Path where to save the generated samples",
            prompt="Path to save the generated samples",
        ),
    ] = "images/my_samples",
):
    """
    Creates random samples from a trained VAE model.
    """

    model, device = load_model(model_path)

    # create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(quantity):
        output = model.sample(device)
        image = utils.convert_tensor_to_image(output)
        sample_name = f"sample_{i+1}.jpg"
        image.save(os.path.join(save_dir, sample_name))


if __name__ == "__main__":
    app()
