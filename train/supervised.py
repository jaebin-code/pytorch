import torch.optim as optim
import torch.nn as nn
from train import get_cifar10_loaders

def supervised(model, device, epochs, batch_size, lr, logger, writer):

    trainloader, testloader = get_cifar10_loaders(batch_size)

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(trainloader) + i)

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)