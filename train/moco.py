import torch
import torch.nn as nn
import torch.nn.functional as F
from train import knn_accuracy, InfoNCE, get_datasets
import torch.optim as optim

def train_moco(model_q, model_k, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, queue_size=8192, m=0.999, T=0.07, epochs=400, device='cuda:1'):
    criterion = InfoNCE(temperature=T)
    global_step = 0

    queue = torch.randn(queue_size, model_q.fc.weight.shape[0]).to(device)
    queue = F.normalize(queue, dim=1)
    queue_ptr = 0

    for epoch in range(epochs):
        model_q.train()
        total_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            im_q, im_k = images[0].to(device), images[1].to(device)

            q = model_q(im_q)
            q = F.normalize(q, dim=1)

            with torch.no_grad():
                for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
                    param_k.data = param_k.data * m + param_q.data * (1.0 - m)

                k = model_k(im_k)
                k = F.normalize(k, dim=1)

            loss = criterion(q, k, queue)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_size = k.shape[0]
                queue[queue_ptr:queue_ptr + batch_size] = k
                queue_ptr = (queue_ptr + batch_size) % queue_size

            total_loss += loss.item()
            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/MoCo', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model_q, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()

def moco(model, model_k, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="MoCo", batch_size=batch_size)

    # Initialize model, optimizer, scheduler
    model_q = model.to(device)
    model_k = model_k.to(device)

    # Copy model_q weights to model_k
    model_k.load_state_dict(model_q.state_dict())

    # Freeze model_k parameters (but keep BN layers active)
    for param in model_k.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model_q.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 400)

    # Train the model with queue and momentum
    train_moco(model_q, model_k, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs=400, device='cuda:1')