# Example (dataset_loader.py)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Loading MNIST dataset...")

def get_dataloaders(dataset='MNIST', batch_size=64):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Add CIFAR-10 handling similarly

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
