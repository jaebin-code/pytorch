import torch.optim as optim
import torch.nn as nn
import torch
from train import get_datasets

def supervised(model, device, epochs, batch_size, lr, wd, momemtum, logger, writer, data_name):
    train_loader, memory_loader, test_loader = get_datasets(data_name=data_name, train_method="Supervised", batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

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
        logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)

        scheduler.step()