import torch
import torch.nn as nn
import torch.optim as optim

from train import get_datasets, rotate_batch, knn_accuracy

def train_rotnet(model, epochs, lr, wd, device, logger, writer, data_name, batch_size):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="RotNet", batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            rotated_inputs, repeated_labels = rotate_batch(inputs, inputs.size(0))
            rotated_inputs, repeated_labels = rotated_inputs.to(device), repeated_labels.to(device)

            optimizer.zero_grad()

            outputs = model(rotated_inputs)
            loss = criterion(outputs, repeated_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('pretrained training loss', loss.item(), epoch * len(train_loader) + i)

        # 각 에포크마다 KNN 정확도 계산
        knn_acc = knn_accuracy(model, memory_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], KNN Accuracy: {knn_acc:.2f}%")
        writer.add_scalar('Accuracy/KNN', knn_acc, epoch)

        scheduler.step()


def rotnet(model, device, num_classes, pretrain_epochs, epochs, batch_size, lr, wd, logger, writer, data_name):
    model.feature_dim = 4
    print(model.feature_dim)
    model = model.to(device)

    print("Training Pretrain...")
    train_rotnet(model, epochs=pretrain_epochs, device=device, lr=lr, wd=wd, logger=logger, writer=writer, data_name=data_name, batch_size=batch_size)

    logger.info('Finished Training')
    writer.close()