import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from data import get_train_loader, get_cifar10_loaders
from models import PretrainedModel, DownstreamModelSecond, DownstreamModelThird, NormalModel
from utils import setup_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_pretrained_model(args):
    trainloader, testloader = get_train_loader(batch_size=args.batch_size)
    model = PretrainedModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    logdir = os.path.join(args.logdir, 'pretrained_training')
    logger, writer = setup_logger(logdir)

    for epoch in range(args.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, start=0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(trainloader) + i)

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images after epoch {epoch + 1}: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)

        print(f'Accuracy of the network on the test images after epoch {epoch + 1}: {accuracy}%')

    print('Finished Training')
    torch.save(model.state_dict(), 'pretrained_rotation_model.pth')
    writer.close()

def train_downstream_second_model(args):
    pretrained_model = PretrainedModel()
    pretrained_model.load_state_dict(torch.load('pretrained_rotation_model.pth'))
    pretrained_model = pretrained_model.to(device)

    downstream_model = DownstreamModelSecond(pretrained_model).to(device)

    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(downstream_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    logdir = os.path.join(args.logdir, 'downstream_second_training')
    logger, writer = setup_logger(logdir)

    for epoch in range(args.downstream_epochs):
        downstream_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = downstream_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(trainloader) + i)

        downstream_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = downstream_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)

        scheduler.step()

    logger.info('Finished Training')
    writer.close()

def train_downstream_third_model(args):
    pretrained_model = PretrainedModel()
    pretrained_model.load_state_dict(torch.load('pretrained_rotation_model.pth'))
    pretrained_model = pretrained_model.to(device)

    downstream_model = DownstreamModelThird(pretrained_model).to(device)
    downstream_model.pretrained_model = pretrained_model

    trainloader, testloader = get_cifar10_loaders(batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(downstream_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    logdir = os.path.join(args.logdir, 'downstream_training')
    logger, writer = setup_logger(logdir)

    for epoch in range(args.downstream_epochs):
        downstream_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = downstream_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(trainloader) + i)

        downstream_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = downstream_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the network on the test images: {accuracy:.2f}%')
        writer.add_scalar('test accuracy', accuracy, epoch)

        scheduler.step()

    logger.info('Finished Training')
    writer.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--downstream_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--logdir', type=str, default='./logs')

    args = parser.parse_args()

    train_pretrained_model(args)
    train_downstream_second_model(args)
    train_downstream_third_model(args)

if __name__ == '__main__':
    main()