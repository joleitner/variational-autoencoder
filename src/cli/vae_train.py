import typer
from src.cli.fully_train import train_fully_connected
from src.cli.conv_train import train_convolutional
from src.cli.vae_resume_train import resume_training

# Subcommand app for VAE training
app = typer.Typer(
    pretty_exceptions_show_locals=False, add_completion=False, help="Train a VAE model"
)

app.command(name="fully")(train_fully_connected)
app.command(name="conv")(train_convolutional)
app.command(name="resume")(resume_training)


if __name__ == "__main__":
    app()
