import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import os
from model import VAE


INITAL_DIM = 784


def load_data(path, batch_size):
    mnist_dataset = torchvision.datasets.MNIST(path,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True)
    return data_loader


def loss_function(x, x_hat, kld):
    # also MSE possible: ((x - x_hat)**2).sum()
    reproduction_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction='sum')
    # kl divergence calculated in the encoder
    return reproduction_loss + kld


def train(model, optimizer, data_loader, epochs):

    for epoch in range(epochs):

        overall_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):

            x = x.view(-1, INITAL_DIM).to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_function(x, x_hat, kld=model.encoder.kld)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ",
              overall_loss / len(data_loader.dataset))

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train VAE on MNIST')

    # Required parameters
    parser.add_argument('--data-path', type=str,
                        help='Path to dataset', required=True)
    parser.add_argument('--batch-size', type=int,
                        default=128, help='Batch size')
    parser.add_argument('--hidden-dim', type=int,
                        default=512, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int,
                        help='Latent dimension', required=True)
    parser.add_argument('--epochs', type=int,
                        default=50, help='Number of epochs')

    parser.add_argument(
        '--save', type=str, default='models/model.pth', help='Path to save model. For example: ./models/model_v1.pth')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = load_data(args.data_path, args.batch_size)

    # calculate input dimension
    INITAL_DIM = data_loader.dataset.data.shape[1] * \
        data_loader.dataset.data.shape[2]

    model = VAE(initial_dim=INITAL_DIM, hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim, device=device).to(device)
    print("Created model: ", model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training starting on", str(device).upper())
    model = train(model, optimizer, data_loader, args.epochs)

    # Save model
    save_dir = os.path.dirname(args.save)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model, args.save)
    print("Model saved to", args.save)
