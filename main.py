import torch
import torch.nn as nn
import torch.optim as optim


import argparse
import os

from train import setup_logger, start_train
from models import get_model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'SVHN', 'STL10'])
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet', 'MLPMixer', 'ConvMixer'])
    parser.add_argument('--train', type=str, default='Supervised', choices=['RotNet', 'SimCLR', 'Supervised', 'MoCo', 'BYOL', 'SimSiam', 'BarlowTwins', 'Separate'])
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_dim', type=int, default=1536)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--feature_dim', type=int, default=128)

    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--downstream_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-6)
    parser.add_argument('--logdir', type=str, default='./logs')

    args = parser.parse_args()

    logdir = os.path.join(args.logdir, f'{args.model}_{args.dataset}_{args.train}')
    logger, writer = setup_logger(logdir)

    device = torch.device("cuda:1") #if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, num_classes=args.num_classes)
    model_k = get_model(args.model, num_classes=args.num_classes)
    start_train(args.train, device, logger=logger, writer=writer, model=model, num_classes=args.num_classes, pretrain_epochs=args.pretrain_epochs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, data_name=args.dataset, temperature=args.temperature, model_k=model_k)

    writer.close()

if __name__ == "__main__":
    main()