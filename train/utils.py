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

class NT_Xent_loss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        N = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        sim_ij = torch.diag(sim, N)
        sim_ji = torch.diag(sim, -N)
        positive_samples = torch.cat([sim_ij, sim_ji], dim=0)
        mask = torch.ones_like(sim)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask.bool()].reshape(2*N, -1)
        labels = torch.zeros(2*N).to(positive_samples.device).long()
        logits = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss

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

class InfoNCE_loss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, keys, queue):
        # query: (N, D)
        # keys: (N, D)
        # queue: (K, D)

        N = query.shape[0]

        l_pos = F.cosine_similarity(query.unsqueeze(1), keys.unsqueeze(0), dim=2) / self.temperature
        l_pos = torch.diag(l_pos).unsqueeze(-1)  # (N, 1)

        l_neg = F.cosine_similarity(query.unsqueeze(1), queue.unsqueeze(0), dim=2) / self.temperature

        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long, device=query.device)

        loss = F.cross_entropy(logits, labels)

        return loss