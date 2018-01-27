from utils.config import config
import logging
import os
import time

import torch


OPTIMIZER_DICT = {'sgd': torch.optim.SGD,  # Gradient Descent
                  'adadelta': torch.optim.Adadelta,  # Adadelta
                  'adagrad': torch.optim.Adagrad,  # Adagrad
                  'adam': torch.optim.Adam,  # Adam
                  'rmsprop': torch.optim.RMSprop  # RMSprop
                  }


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def create_logger(exper=None, file_handler=False, output_dir=None):
    # create logger
    if exper is None and output_dir is None:
        raise ValueError("Parameter -experiment- and -output_dir- cannot be both equal to None")
    logger = logging.getLogger('experiment logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if file_handler:
        if output_dir is None:
            output_dir = exper.output_dir
        fh = logging.FileHandler(os.path.join(output_dir, config.logger_filename))
        # fh.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers

    formatter_ch = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger


def print_flags(exper):
    """
    Prints all entries in argument parser.
    """
    for key, value in vars(exper.run_args).items():
        exper.logger.info(key + ' : ' + str(value))

    if exper.run_args.cuda:
        exper.logger.info(" *** RUNNING ON GPU *** ")


def create_exper_label(exper):

    # retrain = "_retrain" if exper.args.retrain else ""
    exper_label = exper.run_args.model + exper.run_args.version + "_" + str(exper.run_args.epochs) + "E"

    return exper_label
