import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, width, height, patch_size, in_channels, hidden_channels, dropout_rate=0.1):
        super(PatchEmbedding, self).__init__()
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(dropout_rate)

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("The image size must be a multiple of the patch size.")

        self.patch_num = (width // patch_size) * (height // patch_size)

        self.linear = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num + 1, hidden_channels))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.linear(x)  # (batch_size, hidden_channels, num_width, num_height)
        x = x.flatten(2)  # (batch_size, hidden_channels, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, hidden_channels)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # batch_size만큼 확장
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embed  # 포지셔널 인코딩은 배치 상관없이 하나로
        x = self.dropout(x)  # dropout 추가

        return x

class MultiAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout_rate=0.1):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        if hidden_size % num_heads != 0:
            raise ValueError("The hidden size must be a multiple of the num_heads size.")

        self.head_size = hidden_size // num_heads

        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        hidden_channels = x.shape[2]

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        key = key.view(batch_size, num_patches, self.num_heads, self.head_size)
        query = query.view(batch_size, num_patches, self.num_heads, self.head_size)
        value = value.view(batch_size, num_patches, self.num_heads, self.head_size)

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        score_matrix = torch.matmul(query, key.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention_matrix = torch.softmax(score_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)  # dropout 추가
        result_matrix = torch.matmul(attention_matrix, value)

        result_matrix = result_matrix.transpose(1, 2).contiguous().view(batch_size, num_patches, hidden_channels)

        out = self.out(result_matrix)

        return out

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, mlp_dim, dropout_rate=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiAttention(num_heads, hidden_size, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)  # dropout 추가
        )

    def forward(self, x):
        x = self.layer_norm1(x)
        out1 = self.attention(x)
        x = x + out1

        x = self.layer_norm2(x)
        out2 = self.mlp(x)
        x = x + out2

        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, hidden_size=384, num_heads=6, mlp_dim=1596, depth=8, dropout_rate=0.1):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, img_size, patch_size, in_channels, hidden_size, dropout_rate)

        self.attention_blocks = nn.Sequential(
            *[AttentionBlock(num_heads, hidden_size, mlp_dim, dropout_rate) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.attention_blocks(x)
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x