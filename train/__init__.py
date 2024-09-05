from .data import get_datasets, rotate_batch, get_separate_datasets, augment_batch
from .utils import setup_logger, NTXent, InfoNCE, knn_accuracy, loss_byol, loss_simsiam, loss_barlow_twins, loss_separate
from .simclr import simclr
from .supervised import supervised
from .rot import rotnet
from .moco import moco
from .byol import byol
from .simsiam import simsiam
from .barlow_twins import barlow_twins
from .separate import separate

__all__ = ['get_datasets', 'rotate_batch', 'setup_logger', 'NTXent', 'knn_accuracy', 'simclr', 'supervised', 'rotnet', 'InfoNCE', 'loss_separate', 'loss_byol', 'byol', 'loss_simsiam', 'loss_barlow_twins', 'get_separate_datasets', 'augment_batch']

def start_train(name, device, logger, writer, model='ResNet', num_classes=10, pretrain_epochs=50, epochs=100, batch_size=64, lr=1e-3, wd=5e-4, momemtum=0.9, data_name='CIFAR10', temperature=0.5, model_k='ResNet'):
    if name == 'SimCLR':
        simclr(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature)
    elif name == 'Supervised':
        supervised(model, device, epochs, batch_size, lr, wd, momemtum, logger, writer, data_name)
    elif name == 'RotNet':
        rotnet(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name)
    elif name == 'MoCo':
        moco(model, model_k, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature)
    elif name == 'BYOL':
        byol(model, device, epochs, batch_size, lr, wd, logger, writer, data_name)
    elif name == 'SimSiam':
        simsiam(model, device, epochs, batch_size, lr, wd, logger, writer, data_name)
    elif name == 'BarlowTwins':
        barlow_twins(model, device, epochs, batch_size, lr, wd, logger, writer, data_name)
    elif name == 'Separate':
        separate(model, device, epochs, batch_size, lr, wd, logger, writer, data_name)
    else:
        raise ValueError("Unsupported Train Mode")