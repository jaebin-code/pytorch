import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class RotNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        # 원본 이미지와 90도, 180도, 270도 회전된 이미지를 생성
        rot_0 = img
        rot_90 = torch.rot90(img, k=1, dims=[1, 2])
        rot_180 = torch.rot90(img, k=2, dims=[1, 2])
        rot_270 = torch.rot90(img, k=3, dims=[1, 2])

        # 4개의 이미지를 하나의 배치로 만듦
        batch = torch.stack([rot_0, rot_90, rot_180, rot_270])

        # 각 회전에 대한 라벨 (0, 1, 2, 3)
        labels = torch.tensor([0, 1, 2, 3])

        return batch, labels

def get_rot_cifar10_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    rotnet_trainset = RotNetDataset(trainset)
    rotnet_testset = RotNetDataset(testset)

    trainloader = DataLoader(rotnet_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(rotnet_testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_cifar10_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader