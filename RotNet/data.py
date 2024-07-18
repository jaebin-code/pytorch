import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torch

class RotatedDataset(Dataset):
    def __init__(self, original_dataset, rotation_labels):
        self.original_dataset = original_dataset
        self.rotation_labels = rotation_labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return len(self.original_dataset) * len(self.rotation_labels)

    def __getitem__(self, idx):
        original_idx = idx // len(self.rotation_labels)
        rotation_idx = idx % len(self.rotation_labels)
        img, _ = self.original_dataset[original_idx]
        angle = self.rotation_labels[rotation_idx]

        img = img.rotate(angle)

        if self.transform:
            img = self.transform(img)

        return img, rotation_idx

def get_train_loader(batch_size=128, num_workers=2):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    rotation_labels = [0, 90, 180, 270]

    rotated_trainset = RotatedDataset(trainset, rotation_labels)
    rotated_testset = RotatedDataset(testset, rotation_labels)

    trainloader = DataLoader(rotated_trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(rotated_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def get_cifar10_loaders(batch_size=32, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader