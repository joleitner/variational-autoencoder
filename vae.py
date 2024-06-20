import typer

from src.cli.vae_train import app as train_app
from src.cli.vae_generate import generate
from src.cli.vae_sample import sample
from src.cli.vae_dump import dump
from src.cli.vae_info import info


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


app.command(name="generate")(generate)
app.command(name="sample")(sample)
app.command(name="dump")(dump)
app.command(name="info")(info)

app.add_typer(train_app, name="train")


if __name__ == "__main__":
    app()
