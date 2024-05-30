import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from data import get_dataset
from models import get_model
from utils import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN'])
    parser.add_argument('--model', type=str, required=True, choices=['LeNet5', 'LeNet7', 'LeNet9', 'ResNet'])
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--logdir', type=str, default='logs')

    args = parser.parse_args()

    trainloader, testloader = get_dataset(args.dataset, args.batch_size)
    model = get_model(args.model)

    logdir = os.path.join(args.logdir, f'{args.model}_{args.dataset}')
    logger, writer = setup_logger(logdir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(trainloader) + i)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)

        scheduler.step()

    logger.info('Finished Training')
    writer.close()


if __name__ == '__main__':
    main()