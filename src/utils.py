from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import os
import json
import numpy as np
from PIL import Image


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
        json.dump(config, f, indent=2)


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


def load_checkpoint(model_path, device):

    checkpoint = torch.load(
        os.path.join(model_path, "checkpoint.tar"), map_location=torch.device(device)
    )
    return checkpoint


def convert_tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    # detach the tensor from the GPU and convert to numpy array
    image_array = tensor.detach().cpu().numpy()
    # remove batch dimension
    image_array = image_array.squeeze(0)

    image_array = np.uint8(image_array * 255)

    # If the image is grayscale
    if image_array.shape[0] == 1:
        image_array = image_array.squeeze(0)  # Remove channel dimension for grayscale
        mode = "L"

    # Transpose the dimensions from (C, H, W) to (H, W, C) for RGB
    if image_array.ndim == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
        mode = "RGB"

    # Create a PIL image
    return Image.fromarray(image_array, mode=mode)


def image_to_tensor(
    file_path: str, resize: tuple[int, int] = None, grayscale: bool = False
) -> torch.Tensor:
    transform = []
    if resize:
        transform.append(transforms.Resize(resize))
    if grayscale:
        transform.append(transforms.Grayscale())
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    # load image
    image = Image.open(file_path)
    # transform to tensor
    image = transform(image)
    return image
