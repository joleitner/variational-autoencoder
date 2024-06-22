import torch.distributions
from torch.utils.data import DataLoader
import torch
from rich import print
from rich.progress import track

from src.models.vae_base import BaseVAE
import src.utils as utils


def train_model(
    model: BaseVAE,
    data_loader: DataLoader,
    epochs: int,
    checkpoint: dict = None,
    save_path: str = None,
    learning_rate: float = 1e-3,
) -> tuple[BaseVAE, torch.optim.Optimizer, float]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_start = 0

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]

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
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = overall_loss / len(data_loader.dataset)
        print("\tEpoch", epoch + 1 + epoch_start, "\tAverage Loss: {:.4f}".format(loss))

        # when save path is defined save model for each epoch
        if save_path:
            config = utils.load_model_config(save_path)
            config["epochs"] = epoch + 1 + epoch_start
            config["loss_history"].append(loss)
            utils.save_model_config(config, path=save_path)
            utils.save_checkpoint(model, optimizer, epoch + 1 + epoch_start, loss, save_path)

    return model, optimizer, loss
