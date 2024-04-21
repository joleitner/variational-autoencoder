from vae_train import main as train
import typer
from typing_extensions import Annotated
from rich import print


app = typer.Typer()


app.command(name="train")(train)


@app.command()
def test():
    print("Test")


if __name__ == "__main__":
    app()
