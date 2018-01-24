
import torch

from common.parsing import do_parse_args
from common.common import print_flags
from utils.experiment import Experiment
from utils.config import config
from in_out.load_data import HVSMR2016CardiacMRI
from models.model_handler import load_model


def training(exper):
    """

    :param args:
    :return:
    """

    dataset = HVSMR2016CardiacMRI(data_dir=exper.config.data_dir,
                                  search_mask=exper.config.dflt_image_name + ".nii",
                                  norm_scale="rescale", load_type="numpy")

    dcnn_model = load_model(exper)
    del dataset
    del dcnn_model


def main():
    args = do_parse_args()

    exper = Experiment(config, run_args=args)
    exper.start()
    print_flags(exper)

    if args.cmd == 'train':
        training(exper)
    elif args.cmd == 'test':
        raise NotImplementedError("test mode is not yet implemented")


if __name__ == '__main__':
    main()
