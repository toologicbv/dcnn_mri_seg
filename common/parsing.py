import argparse
import sys
import torch


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'], default="train")
    parser.add_argument('-d', '--data-dir', default="/mnt/u_drive/u_share/data/la_sunnybrook/")
    parser.add_argument('-cuda', '--use-cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--bn-sync', action='store_true')

    args = parser.parse_args()
    args.cuda = parser.use_cuda and torch.cuda.is_available()

    assert args.data_dir is not None
    print(' '.join(sys.argv))
    print(args)

    return args

