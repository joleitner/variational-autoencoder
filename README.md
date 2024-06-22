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
- `ìnfo`: Display information about a trained VAE.

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

#### Configuration

After choosing an architecture you will have to configurate your model when running the `train` command. In this example we will train a VAE with a Convolutional architecture on the [Four Shapes](https://www.kaggle.com/datasets/smeschke/four-shapes) dataset. Go ahead and run the following command:

```bash
python vae.py train conv -d data/shapes --resize 100,100 --gray --hiddem_dims 32,64,128,128 --latent_dims 3 --save models/shapes/conv_l3
```

In this example we directly specified the path to the Shapes dataset, resized the images to 100x100 pixels and with the `--gray` option indicated to use the images as grayscale. If you don't specify the resize argument, the images will stay in their original size. Make sure to always set the `--resize / -r` argument when the images in your dataset have different sizes.
Afterwards we specified the **hidden dimensions** of the encoder and decoder as well as the **latent dimensions**. The last argument specifies the folder where the model will be saved.

Afterwards if not specified, you will be prompted to choose the following hyperparameters:

```bash
Batch size for training [32]:
Epochs [30]:
```

You can also just run `python vae.py train conv` and let the CLI prompt you for the required information. Up to your preference.

#### Additional information

A `model_config.json` is saved in the model folder. This file contains all the information about the model configuration. This file is used to load the model later on.
Additionally, a `checkpoint.tar` file is saved after every epoch. This file contains the model state with all trained parameters and the optimizer state to `resume` the training. So in case you would like to stop the training don't worry, your model is saved.

As optimization algorithm the [Adam optimizer](https://arxiv.org/abs/1412.6980) is used with a learning rate of `1e-3`.

### Create random samples

After your training you might want to see what your model has learned. You can generate random samples from the latent space of your trained model. Let's generate 10 random samples:

```bash
python vae.py sample models/shapes/conv_l3 -q 10 --save images/shapes
```

This command will generate 10 random samples from the latent space of the trained model and save them in the `images/shapes` folder.

### Generate an image from a latent vector

In case you have a specific latent vector you would like to generate an image from, you can use the `generate` command. Let's generate an image from a specific latent vector:

```bash
python vae.py generate models/shapes/conv_l3 -v 0.1,-0.6,0.4 --save images/image.jpg
```

This command will generate an image from the latent vector `[0.1, -0.6, 0.4]` and save it as `images/image.jpg`. Additionally, the image will be displayed in a window. In case you don't want to display the image, you can add the `--no-plot` flag.

### Dump the latent space

To analyze the latent space of your trained model, you can dump the latent space of your model to a CSV file. This can later on be loaded as a pandas DataFrame for further analysis. Let's dump the latent space of the trained model:

```bash
python vae.py dump models/shapes/conv_l3 -i data/shapes --save dumps/shapes/latent.csv
```

You specify the model path as argument and with the `-i / --input` option you define which images to use to encode to the latent space. In this case we just dump the whole dataset.
We save the CSV file as `dumps/shapes/latent.csv`.

#### Visualize and analyze the latent space of your model

Inside the `notebooks` folder you will find a Jupyter Notebook [`visualize.ipynb`](notebooks/visualize.ipynb) to visualize and analyze your model's performance and latent space.

In case you trained multiple models, with different architectures or hyperparameters on the same dataset, you can compare them in the [`compare_models.ipynb`](notebooks/compare_models.ipynb) notebook.

### Resume training

In case you realize after your visualization that you would like to train your model for more epochs, you can resume the training with the `train resume` command.

```bash
python vae.py train resume models/shapes/conv_l3 -d data/shapes -e 10
```

This command will resume the training of the model saved in `models/shapes/conv_l3` for 10 additional epochs.

### Display information about a trained model

In case you don't remember the configuration of a trained model, you can display the information using the `info` command.

```bash
python vae.py info models/shapes/conv_l3
```

With the `-d / --detailed` option you can additionally display the detailed model configuration by PyTorch.

## Extend the CLI

In case you feel the need to extend the CLI, for example to add a different VAE Architecture, which may handle other inputs than images. Inside the `src/models`folder you will find a `vae_base.py` file, from which you can extend from. Take an already implemented architecture as an example and modify it to your needs.

Afterwards you simply have to add a new `{architecture}_train.py` file in the `src/cli` folder, which will handle the training of your new architecture. Take the `fully_train.py` file as an example.

You have to add the new architecture to the `cli/vae_train.py` file, so that the CLI can run your newly implemented architecture.

```python
import typer
from src.cli.fully_train import train_fully_connected
from src.cli.conv_train import train_convolutional
# Add your new architecture import here


app = typer.Typer(help="Train a VAE model")

app.command(name="fully")(train_fully_connected)
app.command(name="conv")(train_convolutional)
# Add your new architecture here
app.command(name="new_architecture")(train_new_architecture)
```

That's it. You can now train your new architecture using the CLI and all the other commands will work with it as well.
