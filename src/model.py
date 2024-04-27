import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


class VariationalEncoder(nn.Module):
    def __init__(self, initial_dim: int, hidden_dims: list[int], latent_dim: int, device="cpu"):
        super(VariationalEncoder, self).__init__()

        self.linear1 = nn.Linear(initial_dim, hidden_dims[0])
        # iterate over the hidden_dims and create the hidden layers
        hidden_layers = []
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dim))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.kld = 0
        self.device = device

    def kl_divergence(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.hidden_layers(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        self.kld = self.kl_divergence(mean, logvar)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, initial_dim: int, hidden_dims: list[int], latent_dim: int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dims[-1])
        # iterate reversed over the hidden_dims and create the hidden layers
        hidden_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reversed[1:]):
            hidden_layers.append(nn.Linear(hidden_dims_reversed[i], hidden_dim))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.linear_last = nn.Linear(hidden_dims[0], initial_dim)

    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = self.hidden_layers(x)
        x = torch.sigmoid(self.linear_last(x))
        return x


class VAE(nn.Module):
    def __init__(
        self,
        initial_dim: int = 784,
        hidden_dims: list[int] = [512],
        latent_dim: int = 2,
        device="cpu",
    ):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(initial_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(initial_dim, hidden_dims, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.device = device

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # std deviation
        epsilon = self.N.sample(mean.shape).to(self.device)
        z = mean + std * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat
