from torch.utils.tensorboard import SummaryWriter
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def knn_accuracy(model, memory_data_loader, test_data_loader, device, k=5):
    model.eval()
    total_top1, total_num = 0.0, 0
    feature_bank, feature_labels = [], []

    with torch.no_grad():
        # 메모리 데이터의 특징 추출
        for data, target in memory_data_loader:
            feature = model.extract_features(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).to(device)

        # 테스트 데이터로 kNN 분류 수행
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = model.extract_features(data)
            feature = F.normalize(feature, dim=1)

            # 유사도 계산 및 k-최근접 이웃 찾기
            sim_matrix = torch.mm(feature, feature_bank)
            _, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = feature_labels[sim_indices]

            # 예측 레이블 계산 (가장 빈번한 레이블)
            pred_labels = sim_labels.mode(dim=1)[0]

            total_num += data.size(0)
            total_top1 += (pred_labels == target).sum().item()

    return total_top1 / total_num * 100

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, query, keys, queue):
        query = F.normalize(query, dim=1)
        keys = F.normalize(keys, dim=1)

        # Positive logits
        l_pos = torch.sum(query * keys, dim=1).unsqueeze(-1) # (N, 1)

        # Negative logits
        l_neg = torch.einsum('nc,kc->nk', [query, queue])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(query.device)

        loss = F.cross_entropy(logits, labels)
        return loss

class NTXent(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.einsum('nc,mc->nm', F.normalize(z, dim=-1), F.normalize(z, dim=-1)) / self.temperature

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])

        sim = sim - torch.eye(sim.shape[0], device=sim.device) * 1e9

        loss = F.cross_entropy(sim, labels)

        return loss

def loss_byol(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def loss_simsiam(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return -(x * y).sum(dim=-1)

def loss_barlow_twins(z1, z2, lambda_param=5e-3):
    batch_size = z1.size(0)
    feature_dim = z1.size(1)

    z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
    z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

    cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size

    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(cross_corr).pow_(2).sum()

    loss = on_diag + lambda_param * off_diag
    return loss

def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()