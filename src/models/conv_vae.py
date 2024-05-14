import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


# inspired from: https://www.kaggle.com/code/darkrubiks/variational-autoencoder-with-pytorch#Creating-the-VAE-Model


class ConvVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dims: list[int] = [32, 64, 64],
        image_shape: list[int] = [1, 28, 28],
    ):
        super(ConvVAE, self).__init__()

        self.latent_dims = latent_dim  # Size of the latent space layer
        self.hidden_dims = hidden_dims  # List of hidden layers number of filters/channels
        self.image_shape = image_shape  # Input image shape

        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]
        # Simple formula to get the number of neurons after the last convolution layer is flattened
        self.flattened_channels = int(
            self.last_channels * (self.image_shape[1] / (2 ** len(self.hidden_dims))) ** 2
        )

        # For each hidden layer we will create a Convolution Block
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )

            self.in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Here are our layers for our latent space distribution
        self.mean_layer = nn.Linear(self.flattened_channels, latent_dim)
        self.logvar_layer = nn.Linear(self.flattened_channels, latent_dim)

        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dim, self.flattened_channels)

        # For each Convolution Block created on the Encoder we will do a symmetric Decoder with the same Blocks, but using ConvTranspose
        self.hidden_dims.reverse()
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )

            self.in_channels = h_dim

        self.decoder = nn.Sequential(*modules)

        # The output layer the reconstructed image have the same dimensions as the input image
        self.output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.image_shape[0],
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var componentsbof the latent Gaussian distribution
        mean = self.mean_layer(result)
        log_var = self.logvar_layer(result)

        return [mean, log_var]

    def decode(self, z):
        """
        Maps the given latent codes onto the image space.
        """
        result = self.decoder_input(z)
        result = result.view(
            -1,
            self.last_channels,
            int(self.image_shape[1] / (2 ** len(self.hidden_dims))),
            int(self.image_shape[2] / (2 ** len(self.hidden_dims))),
        )
        result = self.decoder(result)
        result = self.output_layer(result)

        return result

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
        Forward method which will encode and decode our image.
        """
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)

        return [x_hat, mean, log_var, z]

    def loss_function(self, x_hat, x, mean, log_var):
        """
        Computes VAE loss function
        """
        reconstruction_loss = nn.functional.binary_cross_entropy(
            x_hat.reshape(x_hat.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)

        loss = (reconstruction_loss + kld_loss).mean(dim=0)

        return loss

    def sample(self, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(1, self.latent_dims)
        z = z.to(device)
        sample = self.decode(z)

        return sample
