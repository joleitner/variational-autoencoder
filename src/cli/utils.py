import typer
import os


def validate_hidden_dims(hidden_dims: str):
    try:
        valid = [int(item) for item in hidden_dims.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide hidden dimensions as comma-separated integers")
    return hidden_dims


def validate_resize(resize: str):
    if resize is None:
        return None
    try:
        valid = [int(item) for item in resize.split(",")]
        if len(valid) != 2:
            raise ValueError
    except ValueError:
        raise typer.BadParameter(
            "Please provide resize dimensions as comma-separated integers [width, height]"
        )
    return resize


def validate_latent_vector(latent_vector: str):
    try:
        valid = [float(item) for item in latent_vector.split(",")]
    except ValueError:
        raise typer.BadParameter("Please provide latent vector as comma-separated floats")
    return latent_vector


def validate_image_input(input: str):

    if os.path.isfile(input):
        if not input.lower().endswith((".png", ".jpg", ".jpeg")):
            raise typer.BadParameter("Please provide a valid image file")

    if not os.path.exists(input):
        raise typer.BadParameter("Please provide a valid path to an image or directory")

    return input


def convert_resize(resize: str) -> tuple[int, int]:
    return tuple([int(item) for item in resize.split(",")])


def convert_hidden_dims(hidden_dims: str) -> list[int]:
    return [int(item) for item in hidden_dims.split(",")]


def convert_latent_vector(latent_vector: str) -> list[float]:
    return [float(item) for item in latent_vector.split(",")]
