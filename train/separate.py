import torch
import torch.nn as nn
import torch.nn.functional as F
from train import knn_accuracy, loss_separate, get_separate_datasets, augment_batch
import torch.optim as optim

def train_separate(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs, device):
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            augmented_data, augmented_target = augment_batch(inputs, labels)

            outputs = model(augmented_data)
            loss = loss_separate(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 텐서보드에 손실 기록
            writer.add_scalar('Loss/Seperate', loss.item(), global_step)
            global_step += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"accuracy is {accuracy}%")
        writer.add_scalar('test accuracy', accuracy, epoch)
        scheduler.step()

def separate(base_encoder, device, epochs, batch_size, lr, wd, logger, writer, data_name):
    train_loader, memory_loader, test_loader = get_separate_datasets(data_name=data_name, train_method="Seperate", batch_size=batch_size)

    model = base_encoder.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_separate(model, train_loader, memory_loader, test_loader, optimizer, scheduler, writer, epochs=epochs, device=device)