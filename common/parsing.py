import argparse
import sys
import torch

from utils.config import config


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Dilated CNN')
    parser.add_argument('--cmd', choices=['train', 'test'], default="train")
    parser.add_argument('--model', default="dcnn")
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--root_dir', default=config.root_dir)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--bn_sync', action='store_true')
    parser.add_argument('--retrain', action='store_true')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    assert args.root_dir is not None
    return args

