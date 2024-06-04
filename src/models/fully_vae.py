import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from src.models.vae_base import BaseVAE


class FullyVAE(BaseVAE, nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dims: list[int] = [512],
        image_shape: list[int] = [1, 28, 28],
    ):
        super(FullyVAE, self).__init__()

        self.latent_dim: int = latent_dim
        self.hidden_dims: list[int] = hidden_dims
        self.image_shape: list[int] = image_shape
        self.initial_dim = image_shape[0] * image_shape[1] * image_shape[2]

        # Create sequence of hidden layers
        hidden_layers = []
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dim))
            hidden_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(
            nn.Linear(self.initial_dim, hidden_dims[0]), nn.LeakyReLU(), *hidden_layers
        )
        # layers for our latent space distribution
        self.mean_layer = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], self.latent_dim)

        # reverse the hidden_dims and create the decoder
        hidden_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reversed[1:]):
            hidden_layers.append(nn.Linear(hidden_dims_reversed[i], hidden_dim))
            hidden_layers.append(nn.LeakyReLU())

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[-1]),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(hidden_dims[0], self.initial_dim),
            nn.Sigmoid(),
        )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        # Split the result into mu and var components the latent Gaussian distribution
        mean = self.mean_layer(result)
        log_var = self.logvar_layer(result)

        return [mean, log_var]

    def decode(self, z):
        """
        Maps the given latent codes onto the image space.
        """
        x_hat = self.decoder(z)
        return x_hat

    def sample(self, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(1, self.latent_dim).to(device)
        self.eval()
        sample = self.decode(z)
        # reshape the sample to the image shape
        sample = sample.view(-1, *self.image_shape)
        return sample

    def generate(self, z, device):
        """
        Generate an image from the given latent vector
        """
        z = torch.tensor(z).to(device)
        self.eval()
        result = self.decode(z)
        # reshape the result to the image shape
        result = result.view(-1, *self.image_shape)
        return result

    def dump_latent(self, input, device):
        """
        Dumps the latent vector of the input
        """
        input = self.prepare_data(input, device)
        mean, logvar = self.encode(input)
        # remove the batch dimension
        mean, logvar = mean.squeeze(0), logvar.squeeze(0)
        # convert the mean and logvar to numpy
        mean = mean.cpu().detach().numpy()
        std = torch.sqrt(torch.exp(logvar)).cpu().detach().numpy()
        return mean, std

    def prepare_data(self, data, device):
        return data.view(-1, self.initial_dim).to(device)
