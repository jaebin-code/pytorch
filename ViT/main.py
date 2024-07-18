import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from data import get_dataset
from models import get_model
from utils import setup_logger

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN', 'STL10'])
    parser.add_argument('--model', type=str, default='ViT', choices=['ViT', 'ViT_no_CLS', 'ViT_no_POS', 'ViT_no_CLS_POS'])
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_dim', type=int, default=1536)
    parser.add_argument('--depth', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--logdir', type=str, default='./logs')


    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args.dataset, args.batch_size)
    model = get_model(
        args.model,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        depth=args.depth,
    )

    logdir = os.path.join(args.logdir, f'{args.model}_{args.dataset}')
    logger, writer = setup_logger(logdir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.info(f'[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
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
