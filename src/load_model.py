import torch
from typing import Literal

import src.utils as utils
from src.models.conv_vae import ConvVAE
from src.models.fully_vae import FullyVAE
from src.models.vae_base import BaseVAE


def load_model(path: str) -> tuple[BaseVAE, Literal["cuda", "cpu"]]:
    """
    Load the trained model from the given path.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = utils.load_model_config(path)
    checkpoint = utils.load_checkpoint(path)
    if config["model_type"] == "conv":
        model = ConvVAE(
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            image_shape=config["image_shape"],
        )
    elif config["model_type"] == "fully":
        model = FullyVAE(
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            image_shape=config["image_shape"],
        )

    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, device
