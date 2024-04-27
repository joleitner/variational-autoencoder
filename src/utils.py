from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def load_dataset(path, batch_size, resize=None, grayscale=False):

    transform = []
    if resize:
        transform.append(transforms.Resize(resize))
    if grayscale:
        transform.append(transforms.Grayscale())

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    dataset = ImageFolder(root=path, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader
