# Variational Autoencoder

This repository contains a Command Line Interface (CLI) to configure and train a Variational Autoencoder (VAE). The VAE is implemented using [PyTorch](https://pytorch.org/docs/stable/index.html) and the CLI is implemented using [Typer](https://typer.tiangolo.com/).

## Setup your environment

Depending on how you would like to setup your environment, a `requirements.txt` and an `environment.yml` file are provided.

To create a new environment using `conda`, you can run the following command:

```bash
conda env create -f environment.yml
conda activate vae
```

In case you would like to create a virtual environment using `venv`, you can run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, you can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

The entry point of the CLI is the `vae.py` file located directly in the root of the repository. You can run the following command to see the available commands:

```bash
python vae.py --help
```

The CLI provides the following commands:

- `train`: Configure and train a VAE.
- `generate`: Generate an image from a given latent vector.
- `sample`: Sample from the latent space of a trained VAE.
- `dump`: Dump the latent space of a trained VAE.

For every command, you can run `python vae.py <command> --help` to see the available options. Additionally, for most options you will be prompted to choose a value if not specified in the command directly.

### Train a VAE

Let's start by training a new VAE. You have the option to choose in between a **Convolutional** or a **Fully Connected** architecture.
When training with larger images, I recommend using the Convolutional architecture.

Depending on the architecture you either use the `conv`or the `fully` argument.

```bash
# Train a VAE with a Fully Connected architecture
python vae.py train fully
# Train a VAE with a Convolutional architecture
python vae.py train conv
```

#### Dataset

Thanks to the generic data loader `ImageFolder` from Pytorch, you can train on any dataset as long as it is structured in the following way:

```
dataset_name
│
└───class1
│   │   image1.jpg
│   │   image2.jpg
│   │   ...
│
└───class2
    │   image1.jpg
    │   image2.jpg
    │   ...
```

I suggest putting your dataset in the `data` folder. You'll find a [README](/data/README.md) with more information on how and where to download datasets like MNIST or CelebA.

> In case your dataset has no classification, you can put all your images in a single subfolder. For example: `dataset_name/images`.
