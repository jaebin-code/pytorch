import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = self.pointwise(x)
        return x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=4, patch_size=4, n_classes=10):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.layers = nn.ModuleList([ConvMixerLayer(dim, kernel_size) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)