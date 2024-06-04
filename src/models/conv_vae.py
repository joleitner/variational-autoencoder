import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

from src.models.vae_base import BaseVAE


# inspired from: https://www.kaggle.com/code/darkrubiks/variational-autoencoder-with-pytorch#Creating-the-VAE-Model


class ConvVAE(BaseVAE, nn.Module):
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dims: list[int] = [32, 64, 64],
        image_shape: list[int] = [1, 28, 28],
    ):
        super(ConvVAE, self).__init__()

        self.latent_dim: int = latent_dim
        self.hidden_dims: list[int] = hidden_dims
        self.image_shape: list[int] = image_shape  # Input image shape
        self.kernel_size: int = 3  # Kernel size for the convolutions
        self.stride: int = 2  # Stride for the convolutions
        self.padding: int = 1

        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]

        # Calculate the flattend channels after the convolutions
        height, width = self.image_shape[1], self.image_shape[2]

        for _ in self.hidden_dims:
            height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
            width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.flattened_channels = int(self.last_channels * height * width)
        self.conv_output_shape = (self.last_channels, height, width)

        # For each hidden layer we will create a Convolution Block
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )

            self.in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # layers for our latent space distribution
        self.mean_layer = nn.Linear(self.flattened_channels, latent_dim)
        self.logvar_layer = nn.Linear(self.flattened_channels, latent_dim)

        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dim, self.flattened_channels)

        # Reverse hidden dimensions for decoder
        hdim_reversed = hidden_dims.copy()
        hdim_reversed.reverse()

        modules = []
        for i in range(len(hdim_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hdim_reversed[i],
                        out_channels=hdim_reversed[i + 1],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    ),
                    nn.BatchNorm2d(hdim_reversed[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hdim_reversed[-1],
                out_channels=self.image_shape[0],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.Sigmoid(),  # To ensure the output is between 0 and 1
        )

        self.upsample = nn.Upsample(size=self.image_shape[1:], mode="bilinear", align_corners=True)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components the latent Gaussian distribution
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
            *self.conv_output_shape,
        )
        result = self.decoder(result)
        result = self.output_layer(result)
        result = self.upsample(result)  # Ensure the output is the same size as the input

        return result

    def sample(self, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(1, self.latent_dim)
        z = z.to(device)
        print(z)
        self.eval()
        sample = self.decode(z)
        return sample

    def generate(self, z, device):
        """
        Generate an image from the given latent vector
        """
        z = torch.tensor(z).to(device)
        self.eval()
        return self.decode(z)

    def prepare_data(self, data, device):
        """
        Prepare the data for forward pass
        """
        return data.to(device)
