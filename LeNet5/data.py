import torchvision.transforms as transforms
import torchvision
import torch

def get_dataset(name, batch_size):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if name in ["CIFAR10", "CIFAR100", "SVHN"]:
        dataset_class = getattr(torchvision.datasets, name)
        trainset = dataset_class(root='./data', train=True, download=True, transform=transform)
        testset = dataset_class(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader, test_loader
    else:
        raise ValueError(f"Unsupported dataset: {name}")