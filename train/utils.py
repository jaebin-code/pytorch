from torch.utils.tensorboard import SummaryWriter
import logging
import os

import torch.nn as nn

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

class CIFAR10Classifier(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super(CIFAR10Classifier, self).__init__()
        self.features = base_model
        self.classifier = nn.Sequential(
            nn.Linear(2048, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        features = self.features.feature_extract(x)
        out = self.classifier(features)
        return out