from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseVAE(ABC):

    @abstractmethod
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        pass

    @abstractmethod
    def decode(self, z):
        """
        Maps the given latent codes onto the image space.
        """
        pass

    def reparameterization(self, mean, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)  # std deviation
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def forward(self, x):
        """
        Forward method of model to encode and decode our image.
        """
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)

        return [x_hat, mean, log_var, z]

    def loss_function(self, x_hat, x, mean, log_var):
        """
        Computes VAE loss function
        """
        # Mean squared error for grayscale and rgb images
        reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        # Binary cross entropy loss for binary images (black and white)
        # reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        loss = (reconstruction_loss + kld_loss).mean()

        return loss

    @abstractmethod
    def sample(self, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        pass

    @abstractmethod
    def generate(self, z, device):
        """
        Generate an image from the given latent vector
        """
        pass

    @abstractmethod
    def dump_latent(self, input, device):
        """
        Dumps the latent vector of the input
        """
        pass

    @abstractmethod
    def prepare_data(self, data, device):
        """
        Prepare the data for forward pass
        """
        pass
