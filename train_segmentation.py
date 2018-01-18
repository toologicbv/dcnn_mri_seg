
import json
import logging
import math
import os
import pdb
from os.path import exists, join, split

from common.parsing import parse_args


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def training(args):
    """

    :param args:
    :return:
    """

    pass


def main():
    args = parse_args()
    if args.cmd == 'train':
        training(args)
    elif args.cmd == 'test':
        raise NotImplementedError("test mode is not yet implemented")


if __name__ == '__main__':
    main()
