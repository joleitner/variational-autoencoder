import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu'):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.kld = 0
        self.device = device

    def kl_divergence(self, mean, log_var):
        return - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        self.kld = self.kl_divergence(mean, logvar)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, initial_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, initial_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VAE(nn.Module):
    def __init__(self, initial_dim=784, hidden_dim=512, latent_dim=2, device='cpu'):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(
            initial_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(initial_dim, hidden_dim, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.device = device

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # std deviation
        epsilon = self.N.sample(mean.shape).to(self.device)
        z = mean + std*epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat
