import torch
import torch.nn as nn
import torch.nn.functional as F
from train import knn_accuracy, InfoNCE_loss, get_datasets
import torch.optim as optim

def train_moco(model, model_k, train_loader, optimizer, criterion, scheduler, epochs, device, writer, memory_loader, test_loader, queue_size=32768, m=0.999):
    model.train()
    model_k.eval()
    global_step = 0

    queue = torch.randn(queue_size, model.feature_dim).to(device)
    queue_ptr = 0

    for epoch in range(epochs):
        for (x_q, x_k), _ in train_loader:
            x_q, x_k = x_q.to(device), x_k.to(device)

            q = model(x_q)

            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), model_k.parameters()):
                     param_k.data = param_k.data * m + param_q.data * (1 - m)

                k = model_k(x_k)

            loss = criterion(q, k, queue)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_size = k.shape[0]
                queue[queue_ptr:queue_ptr + batch_size] = k.detach()
                queue_ptr = (queue_ptr + batch_size) % queue_size

            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/MoCo', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model_k, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()

    return model_k

def moco(model, model_k, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="MoCo", batch_size=batch_size)

    model = model.to(device)
    model_k = model_k.to(device)

    for param_q, param_k in zip(model.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = InfoNCE_loss(temperature).to(device)

    train_moco(model, model_k, train_loader, optimizer, criterion, scheduler, epochs, device=device, writer=writer, memory_loader=memory_loader, test_loader=test_loader, queue_size=32768, m=0.999)