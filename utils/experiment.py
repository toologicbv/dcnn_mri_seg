import torch
import numpy as np
import os
from datetime import datetime
from pytz import timezone
import dill
from common.parsing import create_def_argparser, run_dict

from common.common import create_logger, create_exper_label


class Experiment(object):

    def __init__(self, config, run_args=None, set_seed=True):

        # logging
        self.epoch_id = 0
        self.chkpnt_dir = None
        self.logger = None
        self.output_dir = None
        self.optimizer = None
        # set this later
        self.batches_per_epoch = 0
        if run_args is None:
            self.run_args = create_def_argparser(**run_dict)
        else:
            self.run_args = run_args
        self.config = config
        self.epoch_stats = None
        self.val_stats = None
        self._set_path()
        self.init_statistics()

        if set_seed:
            SEED = 2345
            torch.manual_seed(SEED)
            if run_args.cuda:
                torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)

    def init_statistics(self):
        val_runs = (self.run_args.epochs // self.run_args.val_freq) + 1
        self.epoch_stats = {'mean_loss': np.zeros(self.run_args.epochs)}
        self.val_stats = {'mean_loss': np.zeros(val_runs)}

    def start(self, exper_logger=None):

        if exper_logger is None:
            self.logger = create_logger(self, file_handler=True)
        else:
            self.logger = exper_logger

    def _set_path(self):
        if self.run_args.log_dir is None:
            self.run_args.log_dir = str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                    '%Y-%m-%d %H:%M:%S.%f')[:-7], ' ', '_') + "_" + create_exper_label(self) + \
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
            if self.run_args.chkpnt:
                self.chkpnt_dir = os.path.join(log_dir, self.config.checkpoint_path)
                os.makedirs(self.chkpnt_dir)
        self.output_dir = log_dir

    def save(self, file_name=None):
        if file_name is None:
            file_name = "exper_stats@{}".format(self.epoch_id) + ".dll"

        outfile = os.path.join(self.output_dir, file_name)
        logger = self.logger
        optimizer = self.optimizer
        self.logger = None
        self.optimizer = None

        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        self.logger = logger
        self.optimizer = optimizer

        if self.logger is not None:
            self.logger.info("Epoch: {} - Saving experimental details to {}".format(self.epoch_id, outfile))
