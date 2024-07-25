import torch
import torch.nn as nn
import torch.optim as optim

from train import get_cifar10_loaders, get_rot_cifar10_loaders, CIFAR10Classifier

def train(model, trainloader, testloader, epochs, lr, device, logger, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # (batch_size * 4, 3, 32, 32)
            inputs = inputs.view(-1, 3, 32, 32)
            # (batch_size * 4,)
            labels = labels.view(-1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('pretrained training loss', loss.item(), epoch * len(trainloader) + i)

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # (batch_size * 4, 3, 32, 32)
                inputs = inputs.view(-1, 3, 32, 32)
                # (batch_size * 4,)
                labels = labels.view(-1)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images after epoch {epoch + 1}: {accuracy:.2f}%')
        writer.add_scalar('pretrained test accuracy', accuracy, epoch)


def rot_train(model, trainloader, testloader, epochs, lr, device, logger, writer):
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

def rotnet(model, device, pretrain_epochs, epochs, batch_size, lr, logger, writer):
    # Load data for RotNet
    trainloader, testloader = get_rot_cifar10_loaders(batch_size)

    # Initialize models
    model = model
    rotnet = CIFAR10Classifier(model, num_classes=4).to(device)

    # Train RotNet
    print("Training Pretrain...")
    train(rotnet, trainloader, testloader, epochs=pretrain_epochs, device=device, lr=lr, logger=logger, writer=writer)

    # Freeze ResNet parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load data for CIFAR-10 classification
    trainloader, testloader = get_cifar10_loaders(batch_size)

    # Use pre-trained ResNet for CIFAR-10 classification
    cifar10_classifier = CIFAR10Classifier(model).to(device)

    print("Training Classifer...")
    rot_train(cifar10_classifier, trainloader, testloader, epochs=epochs, device=device, lr=lr, logger=logger, writer=writer)

    logger.info('Finished Training')
    writer.close()