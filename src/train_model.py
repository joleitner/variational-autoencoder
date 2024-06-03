import torch.distributions
from torch.utils.data import DataLoader
import torch
from rich import print
from rich.progress import track

from src.models.vae_base import BaseVAE


def train_model(
    model: BaseVAE,
    data_loader: DataLoader,
    epochs: int,
) -> tuple[BaseVAE, torch.optim.Optimizer, float]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    for epoch in track(
        range(epochs), description="Training process"
    ):  # track is showing a progress bar

        model.train()
        overall_loss = 0

        for batch_idx, (x, _) in enumerate(data_loader):

            x = model.prepare_data(x, device)

            optimizer.zero_grad()
            x_hat, mean, log_var, _ = model(x)
            loss = model.loss_function(x_hat, x, mean, log_var)
            overall_loss += loss
            loss.backward()
            optimizer.step()

        loss = overall_loss / batch_idx
        print("\tEpoch", epoch + 1, "\tAverage Loss: {:.4f}".format(loss))

    return model, optimizer, loss
