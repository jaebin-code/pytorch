from .ViT import ViT
from .ResNet import ResNet50, Bottleneck, ResNet18

def get_model(name, img_size=32, patch_size=4, in_channels=3, num_classes=10, hidden_size=384, num_heads=6, mlp_dim=1596, depth=8, feature_dim=128):
    if name == "ViT":
        return ViT(img_size, patch_size, in_channels, num_classes, hidden_size, num_heads, mlp_dim, depth)
    elif name == "ResNet":
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError("Unsupported Model")


__all__ = ['ResNet', 'ViT', 'get_model']