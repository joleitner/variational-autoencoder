from abc import ABC, abstractmethod


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

    @abstractmethod
    def reparameterization(self, mean, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Forward method of model to encode and decode.
        """
        pass

    @abstractmethod
    def loss_function(self, x_hat, x, mean, log_var):
        """
        Computes VAE loss function
        """
        pass

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
    def prepare_data(self, data, device):
        """
        Prepare the data for forward pass
        """
        pass
