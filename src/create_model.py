import torch
from typing import Literal

from src.models.conv_vae import ConvVAE
from src.models.fully_vae import FullyVAE
from src.models.vae_base import BaseVAE


def create_vae_model(
    model_type: Literal["fully", "conv"],
    latent_dim: int,
    hidden_dims: list[int],
    image_shape: list[int],
) -> tuple[BaseVAE, Literal["cuda", "cpu"]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "conv":
        model = ConvVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, image_shape=image_shape)
    elif model_type == "fully":
        model = FullyVAE(latent_dim=latent_dim, hidden_dims=hidden_dims, image_shape=image_shape)
    else:
        raise ValueError(
            "Model type not recognized. Please choose from 'fully_connected' or 'convolutional'"
        )

    return model.to(device), device
