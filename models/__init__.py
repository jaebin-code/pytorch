from .ViT import ViT
from .ResNet import ResNet50, Bottleneck, ResNet18
from .MLPMixer import MLPMixer
from .ConvMixer import ConvMixer

def get_model(name, img_size=32, patch_size=4, in_channels=3, num_classes=10, hidden_size=384, num_heads=6, mlp_dim=1596, depth=8, feature_dim=128):
    if name == "ViT":
        return ViT(img_size, patch_size, in_channels, num_classes, hidden_size, num_heads, mlp_dim, depth)
    elif name == "ResNet":
        return ResNet18(num_classes=num_classes)
    elif name == "MLPMixer":
        return MLPMixer(width=img_size, height=img_size, patch_size=patch_size, in_channels=in_channels, hidden_dim=hidden_size, token_mixing_dim=384, channel_mixing_dim=1536, num_classes=10, depth=4)
    elif name == "ConvMixer":
        return ConvMixer(dim=hidden_size, depth=4, kernel_size=9, patch_size=7, n_classes=10)
    else:
        raise ValueError("Unsupported Model")


__all__ = ['ResNet', 'ViT', 'get_model']