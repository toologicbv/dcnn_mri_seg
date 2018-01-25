import time
import torch

from common.parsing import do_parse_args
from common.common import print_flags, OPTIMIZER_DICT
from utils.experiment import Experiment
from utils.config import config
from utils.batch_handlers import TwoDimBatchHandler
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
    exper.optimizer = OPTIMIZER_DICT[exper.config.optimizer](dcnn_model.parameters(), lr=exper.run_args.lr)

    exper.batches_per_epoch = (dataset.__len__() / exper.run_args.batch_size)
    exper.logger.info("Size data-set {} // number of epochs {} // batch-size {} // batches/epoch {}".format(
        dataset.__len__(), exper.run_args.epochs, exper.run_args.batch_size, exper.batches_per_epoch))

    for epoch_id in range(exper.run_args.epochs):
        start_time = time.time()
        exper.logger.info("Start epoch {}".format(epoch_id+1))
        for batch_id in range(exper.batches_per_epoch):
            new_batch = TwoDimBatchHandler(exper)
            new_batch.generate_batch_2d(dataset.images, dataset.labels)
            # print("new_batch.b_images", new_batch.b_images.shape)
            b_out, b_out_softmax = dcnn_model(new_batch.b_images)
            b_loss = dcnn_model.get_loss(b_out, new_batch.b_labels)
            # compute gradients w.r.t. model parameters
            b_loss.backward(retain_graph=False)
            exper.optimizer.step()
            sum_grads = dcnn_model.sum_grads()
            exper.logger.info("Current batch-loss {:.4f} / gradients {:.3f}".format(
                b_loss.data.cpu().squeeze().numpy()[0], sum_grads))
            dcnn_model.zero_grad()

        end_time = time.time()
        total_time = end_time - start_time
        exper.logger.info("End epoch {}: duration {:.2f}".format(epoch_id + 1, total_time))
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
