import torch.nn as nn
import torch.nn.functional as F

class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NINBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.block2 = NINBlock(96, 192)
        self.block3 = NINBlock(192, 192)

        self.fc1 = nn.Linear(192 * 4 * 4, 200)
        self.bn_fc1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn_fc2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 4)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

class DownstreamModelSecond(nn.Module):
    def __init__(self, pretrained_model):
        super(DownstreamModelSecond, self).__init__()
        self.block1 = pretrained_model.block1
        self.block2 = pretrained_model.block2

        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(192 * 8 * 8, 10)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class DownstreamModelThird(nn.Module):
    def __init__(self, pretrained_model):
        super(DownstreamModelThird, self).__init__()
        self.block1 = pretrained_model.block1
        self.block2 = pretrained_model.block2
        self.block3 = pretrained_model.block3

        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        for param in self.block3.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(192 * 4 * 4, 10)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

class NormalModel(nn.Module):
    def __init__(self):
        super(NormalModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.block2 = NINBlock(96, 192)
        self.block3 = NINBlock(192, 192)

        self.fc1 = nn.Linear(192 * 4 * 4, 200)
        self.bn_fc1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn_fc2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

def get_model(name):
    if name == "Pretrained":
        return PretrainedModel()
    elif name == "DownstreamSecond":
        return DownstreamModelSecond()
    elif name == "DownstreamThird":
        return DownstreamModelThird()
    elif name == "NormalModel":
        return NormalModel()
    else:
        raise ValueError("Unsupported Model")