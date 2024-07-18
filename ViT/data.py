import torchvision.transforms as transforms
import torchvision
import torch

def get_dataset(name, batch_size):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop((32, 32)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.491, 0.482 ,0.446), (0.247,0.243, 0.261))]
    )

    if name in ["CIFAR10", "CIFAR100"]:
        dataset_class = getattr(torchvision.datasets, name)
        trainset = dataset_class(root='./data', train=True, download=True, transform=transform)
        testset = dataset_class(root='./data', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

        return trainloader, testloader
    elif name in ["SVHN"]:
        dataset_class = getattr(torchvision.datasets, name)
        trainset = dataset_class(root='./data', split='train', download=True, transform=transform)
        testset = dataset_class(root='./data', split='test', download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

        return trainloader, testloader
    elif name in ["STL10"]:
        dataset_class = getattr(torchvision.datasets, name)
        trainset = dataset_class(root='./data', split='train', download=True, transform=transform)
        testset = dataset_class(root='./data', split='test', download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        return trainloader, testloader
    else:
        raise ValueError(f"Unsupported dataset: {name}")
