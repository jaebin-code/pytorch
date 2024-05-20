import argparse
import torch
import os
from data import get_dataset
from models import get_model
from utils import setup_logger, train_and_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN'])
    parser.add_argument('--model', type=str, required=True, choices=['LeNet5', 'LeNet7', 'LeNet9'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--logdir', type=str, default='logs')

    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args.dataset, args.batch_size)
    model = get_model(args.model)

    logdir = os.path.join(args.logdir, f'{args.model}_{args.dataset}')
    logger, writer = setup_logger(logdir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_and_evaluate(model, train_loader, test_loader, device, writer, logger, args.epochs, args.lr, args.momentum, args.weight_decay)

if __name__ == '__main__':
    main()