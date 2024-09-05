import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMixerLayer(nn.Module):
    def __init__(self, patch_num, hidden_dim, token_mixing_dim, channel_mixing_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(patch_num, token_mixing_dim),
            nn.GELU(),
            nn.Linear(token_mixing_dim, patch_num)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channel_mixing_dim),
            nn.GELU(),
            nn.Linear(channel_mixing_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x + self.token_mixing(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixing(x)

        return x

class MLPMixer(nn.Module):
    def __init__(self, width, height, patch_size, in_channels, hidden_dim, token_mixing_dim, channel_mixing_dim, num_classes, depth):
        super().__init__()

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("The image size must be a multiple of the patch size.")

        patch_num = (width // patch_size) * (height // patch_size)
        self.embedding_patch = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        # 깊이 쌓는 코드 알아두기
        self.layers = nn.ModuleList({
            MLPMixerLayer(patch_num, hidden_dim, token_mixing_dim, channel_mixing_dim)
            for _ in range(depth)
        })
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.embedding_patch(x).flatten(2).transpose(1, 2) # (batch, channel, patch) -> (batch, patch, channel)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.fc(x)