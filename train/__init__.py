from .data import RotNetDataset, get_cifar10_loaders, get_rot_cifar10_loaders
from .utils import setup_logger, CIFAR10Classifier
from .rot import train, rot_train, rotnet
from .supervised import supervised

__all__ = ['RotNetDataset', 'get_cifar10_loaders', 'get_rot_cifar10_loaders', 'setup_logger', 'CIFAR10Classifier', 'train', 'rot_train', 'rotnet', 'supervised']

def start_train(name, device, logger, writer, model="ResNet", pretrain_epochs=50, epochs=100, batch_size=64, lr=1e-3):
    if name == "RotNet":
        rotnet(model, device, pretrain_epochs, epochs, batch_size, lr, logger, writer)
    elif name == "Supervised":
        supervised(model, device, epochs, batch_size, lr, logger, writer)
    else:
        raise ValueError("Unsupported Train Mode")