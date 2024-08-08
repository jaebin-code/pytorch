from train import get_datasets, knn_accuracy, NT_Xent_loss
import torch.optim as optim

def train_simclr(model, train_loader, optimizer, criterion, scheduler, epochs, device, writer, memory_loader, test_loader):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        for (x_i, x_j), _ in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j = model(x_i), model(x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/SimCLR', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()

    return model

def simclr(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name, temperature):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="SimCLR", batch_size=batch_size)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = NT_Xent_loss(temperature).to(device)

    train_simclr(model, train_loader, optimizer, criterion, scheduler, epochs, device, writer, memory_loader, test_loader)