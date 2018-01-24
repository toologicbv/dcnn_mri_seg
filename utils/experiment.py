import torch
import numpy as np
import argparse
import os
from datetime import datetime
from pytz import timezone

from config import config

from common.common import create_logger, create_exper_label


run_dict = {'cmd': 'train',
            'data_dir': config.data_dir,
            'use_cuda': True,
            'epochs': 10,
            'batch_size': 10,
            'lr': 1e-4
}


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.cmd = kwargs['cmd']
    args.data_dir = kwargs['data_dir']
    args.use_cuda = kwargs['use_cuda']
    args.epochs = kwargs['epochs']
    args.batch_size = kwargs['batch_size']
    args.lr = kwargs['lr']
    args.retrain = kwargs['retrain']
    args.cuda = args.use_cuda and torch.cuda.is_available()
    return args


class Experiment(object):

    def __init__(self, config, run_args=None, set_seed=False):

        # logging
        self.logger = None
        self.output_dir = None

        if run_args is None:
            self.run_args = create_def_argparser(**run_dict)
        else:
            self.run_args = run_args
        self.config = config
        self._set_path()

        if set_seed:
            SEED = 2345
            torch.manual_seed(SEED)
            if run_args.cuda:
                torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)

    def start(self, exper_logger=None):

        if exper_logger is None:
            self.logger = create_logger(self, file_handler=True)
        else:
            self.logger = exper_logger

    def _set_path(self):
        if self.run_args.log_dir is None:
            self.run_args.log_dir = self.run_args.model + + "_" + \
                                str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                    '%Y-%m-%d %H:%M:%S.%f')[:-7],
                                            ' ', '_') + "_" + create_exper_label(self) + \
                                "_lr" + "{:.0e}".format(self.run_args.lr)
            self.run_args.log_dir = str.replace(str.replace(self.run_args.log_dir, ':', '_'), '-', '')

        else:
            # custom log dir
            self.run_args.log_dir = str.replace(self.run_args.log_dir, ' ', '_')
            self.run_args.log_dir = str.replace(str.replace(self.run_args.log_dir, ':', '_'), '-', '')
        log_dir = os.path.join(self.config.log_root_path, self.run_args.log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            fig_path = os.path.join(log_dir, self.config.figure_path)
            os.makedirs(fig_path)

        self.output_dir = log_dir
