import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import os

def setup_logger(logdir):
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logdir, 'log.txt')),
            logging.StreamHandler(os.sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    writer = SummaryWriter(logdir)

    return logger, writer