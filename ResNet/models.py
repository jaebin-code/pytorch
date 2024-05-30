import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

class LeNet7(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(32*3*3, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

class LeNet9(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 16, 5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x):
        return nn.ReLU(self.seq(x) + self.shortcut(x))




class ResNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super().__init__(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

def get_model(name):
    if name == "LeNet5":
        return LeNet5()
    elif name == "LeNet7":
        return LeNet7()
    elif name == "LeNet9":
        return LeNet9()
    elif name == "ResNet":
        return ResNet()
    else:
        raise ValueError("Unsupported model")