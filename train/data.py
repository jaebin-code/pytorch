import torchvision
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader

def get_train_normalize_transform(dataset_name):
    if dataset_name == "CIFAR10":
        return transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.RandomResizedCrop(32, antialias=True),
        ])
    elif dataset_name == "CIFAR100":
        return transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_test_normalize_transform(dataset_name):
    if dataset_name == "CIFAR10":
        return transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Resize(32, antialias=True),
        ])
    elif dataset_name == "CIFAR100":
        return transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_base_transform(train_method='Supervised'):
    if train_method == 'Supervised' or train_method == 'RotNet':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif train_method == 'SimCLR' or train_method == 'MoCo':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])


class TwoCropTransform:
    def __init__(self, base_transform, normalize_transform):
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [self.normalize_transform(q), self.normalize_transform(k)]

def rotate_batch(batch, batch_size):
    # batch shape: (B, C, H, W)
    B, C, H, W = batch.shape

    # 4방향으로 회전된 이미지를 저장할 텐서
    rotated_batch = torch.zeros(B * 4, C, H, W)

    # 각 회전된 배치를 한 번에 생성
    rotated_batch[0::4] = batch  # 0도 회전
    rotated_batch[1::4] = torch.rot90(batch, k=1, dims=(2, 3))  # 90도 회전
    rotated_batch[2::4] = torch.rot90(batch, k=2, dims=(2, 3))  # 180도 회전
    rotated_batch[3::4] = torch.rot90(batch, k=3, dims=(2, 3))  # 270도 회전
    labels = torch.tensor([0, 1, 2, 3] * batch_size, device='cpu')

    return rotated_batch, labels


def get_datasets(data_name="CIFAR10", train_method="Supervised", batch_size=64):
    if data_name not in ["CIFAR10", "CIFAR100"]:
        raise ValueError(f"Unsupported dataset: {data_name}")

    normalize_transform = get_train_normalize_transform(data_name)
    test_normalize_transform = get_test_normalize_transform(data_name)
    train_base_transform = get_base_transform(train_method)
    test_base_transform = get_base_transform(train_method='test')

    if train_method == 'SimCLR' or train_method == 'MoCo':
        train_transform = TwoCropTransform(train_base_transform, normalize_transform)
    else:
        train_transform = transforms.Compose([train_base_transform, normalize_transform])
    test_transform = transforms.Compose([test_base_transform, test_normalize_transform])

    dataset_class = getattr(torchvision.datasets, data_name)

    train_dataset = dataset_class(root='./data', train=True, download=True, transform=train_transform)
    memory_dataset = dataset_class(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, memory_loader, test_loader