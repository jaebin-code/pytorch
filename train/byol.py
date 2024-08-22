import torch
import torch.nn as nn
import torch.nn.functional as F
from train import knn_accuracy, loss_byol, get_datasets
import torch.optim as optim
import copy

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, hidden_dim=2048):
        super(BYOL, self).__init__()
        self.online_encoder = base_encoder
        self.target_encoder = copy.deepcopy(base_encoder)

        # Projection MLP
        self.online_projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.target_projector = copy.deepcopy(self.online_projector)

        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        online_proj1 = self.online_projector(self.online_encoder(x1))
        online_proj2 = self.online_projector(self.online_encoder(x2))

        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)

        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder(x1))
            target_proj2 = self.target_projector(self.target_encoder(x2))

        return online_pred1, online_pred2, target_proj1.detach(), target_proj2.detach()

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

def train_byol(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs, device):
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_loader):
            x1, x2 = images[0].to(device), images[1].to(device)

            online_pred1, online_pred2, target_proj1, target_proj2 = model(x1, x2)

            loss = (
                loss_byol(online_pred1, target_proj2.detach()) +
                loss_byol(online_pred2, target_proj1.detach())
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target()

            total_loss += loss.item()
            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/BYOL', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model.online_encoder, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()

def byol(base_encoder, device, epochs, batch_size, lr, wd, logger, writer, data_name):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="BYOL", batch_size=batch_size)

    base_encoder = base_encoder
    base_encoder.fc = nn.Identity()
    model = BYOL(base_encoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_byol(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs=epochs, device=device)