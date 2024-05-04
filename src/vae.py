from vae_train import train
from vae_generate import generate
from vae_dump import dump
import typer
from typing_extensions import Annotated
from rich import print


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


app.command(name="train")(train)
app.command(name="generate")(generate)
app.command(name="dump")(dump)


if __name__ == "__main__":
    app()
