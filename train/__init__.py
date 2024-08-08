from .data import get_datasets, rotate_batch
from .utils import setup_logger, NT_Xent_loss, InfoNCE_loss, knn_accuracy
from .simclr import simclr
from .supervised import supervised
from .rot import rotnet
from .moco import moco


__all__ = ['get_datasets', 'rotate_batch', 'setup_logger', 'NT_Xent_loss', 'knn_accuracy', 'simclr', 'supervised', 'rotnet', 'InfoNCE_loss']

def start_train(name, device, logger, writer, model='ResNet', num_classes=10, pretrain_epochs=50, epochs=100, batch_size=64, lr=1e-3, wd=5e-4, momemtum=0.9, data_name='CIFAR10', temperature=0.5, model_k='ResNet'):
    if name == 'SimCLR':
        simclr(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature)
    elif name == 'Supervised':
        supervised(model, device, epochs, batch_size, lr, wd, momemtum, logger, writer, data_name)
    elif name == 'RotNet':
        rotnet(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name)
    elif name == 'MoCo':
        moco(model, model_k, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature)
    else:
        raise ValueError("Unsupported Train Mode")