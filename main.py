import argparse
from tkinter import N
import finetune
import pretrain
import torch
import random
import numpy as np
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pretrain.train(**vars(args))
    finetune.train(**vars(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='resnet32-all')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--teacher-model', type=str, default='resnet32')
    parser.add_argument('--dataset-path', type=str, default='../data/vision/cifar10/')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--channel-size-lst', nargs='+', type=int, default=[16, 32, 64])
    parser.add_argument('--div-indices', nargs='+', type=int, default=[5, 10, 15])
    parser.add_argument('--start-index', type=int, default=None)
    parser.add_argument('--end-index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warm-up-epoch', type=int, default=5)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--mlp-dim', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    main(args)

