import torch
import torch.nn as nn
import torch.nn.functional as F
from train import knn_accuracy, loss_simsiam, get_datasets
import torch.optim as optim

class SimSiam(nn.Module):
    def __init__(self, base_encoder, projection_dim=512, hidden_dim=2048):
        super(SimSiam, self).__init__()
        self.encoder = base_encoder

        self.predictor = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x1, x2):
        encoding1 = self.encoder(x1)
        encoding2 = self.encoder(x2)

        pred1 = self.predictor(encoding1)
        pred2 = self.predictor(encoding2)

        return pred1, encoding2.detach(), pred2, encoding1.detach()

def train_simsiam(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs, device):
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_loader):
            x1, x2 = images[0].to(device), images[1].to(device)

            pred1, encoding2, pred2, encoding1 = model(x1, x2)

            loss = (
                loss_simsiam(pred1, encoding2.detach()) +
                loss_simsiam(pred2, encoding1.detach())
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/SimSiam', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model.encoder, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()

def simsiam(base_encoder, device, epochs, batch_size, lr, wd, logger, writer, data_name):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="SimSiam", batch_size=batch_size)

    base_encoder = base_encoder
    base_encoder.fc = nn.Identity()
    model = SimSiam(base_encoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_simsiam(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs=epochs, device=device)

