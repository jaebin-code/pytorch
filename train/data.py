import torchvision
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

class TwoCropTransform:
    def __init__(self, base_transform, normalize_transform):
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [self.normalize_transform(q), self.normalize_transform(k)]

def get_cifar10_transforms(train_method='Supervised'):
    normalize_transform = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    if train_method in ['Supervised', 'RotNet']:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif train_method in ['SimCLR', 'MoCo']:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])
    else:
        base_transform = transforms.Compose([
            transforms.Resize(32, antialias=True),
            transforms.ToTensor()
        ])

    if train_method in ['SimCLR', 'MoCo']:
        train_transform = TwoCropTransform(base_transform, normalize_transform)
    else:
        train_transform = transforms.Compose([base_transform, normalize_transform])

    test_transform = transforms.Compose([
        transforms.Resize(32, antialias=True),
        transforms.ToTensor(),
        normalize_transform
    ])

    return train_transform, test_transform

def rotate_batch(batch, batch_size):
    B, C, H, W = batch.shape
    rotated_batch = torch.zeros(B * 4, C, H, W)
    rotated_batch[0::4] = batch
    rotated_batch[1::4] = torch.rot90(batch, k=1, dims=(2, 3))
    rotated_batch[2::4] = torch.rot90(batch, k=2, dims=(2, 3))
    rotated_batch[3::4] = torch.rot90(batch, k=3, dims=(2, 3))
    labels = torch.tensor([0, 1, 2, 3] * batch_size, device='cpu')
    return rotated_batch, labels

class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            im_q = self.transform(image)
            im_k = self.transform(image)
        return [im_q, im_k], label

def get_datasets(data_name='CIFAR10', train_method='Supervised', batch_size=64):
    # Image augmentations
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


    # Dataset and Dataloader
    train_dataset = CustomCIFAR10(root='./data', train=True, download=True, transform=transform1)
    memory_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, memory_loader, test_loader