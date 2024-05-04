from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import os
import json


def load_dataset(path, batch_size, resize=None, grayscale=False, shuffle=True):

    transform = []
    if resize:
        transform.append(transforms.Resize(resize))
    if grayscale:
        transform.append(transforms.Grayscale())

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    dataset = ImageFolder(root=path, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def save_model_config(config, path):
    file_name = "model_config.json"
    file_path = os.path.join(path, file_name)
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "w") as f:
        json.dump(config, f)


def load_model_config(model_path):
    with open(os.path.join(model_path, "model_config.json"), "r") as f:
        config = json.load(f)
    return config


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    file_name = "checkpoint.tar"
    file_path = os.path.join(path, file_name)
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(checkpoint, file_path)


def load_checkpoint(model_path):
    checkpoint = torch.load(os.path.join(model_path, "checkpoint.tar"))
    return checkpoint
