import time
import os
import glob
import torch
from torch.autograd import Variable
import numpy as np

from common.parsing import do_parse_args
from common.common import print_flags, OPTIMIZER_DICT
from utils.experiment import Experiment
from utils.config import config
from utils.batch_handlers import TwoDimBatchHandler
from in_out.load_data import HVSMR2016CardiacMRI
from models.model_handler import load_model, save_checkpoint

from jelmer_util.helper import loadImageDir, generateBatch2D


def training(exper):
    """

    :param args:
    :return:
    """

    dataset = HVSMR2016CardiacMRI(data_dir=exper.config.data_dir,
                                  search_mask=exper.config.dflt_image_name + ".nii",
                                  norm_scale=exper.config.norm_method, load_type="numpy",
                                  val_fold=exper.run_args.val_fold_id)

    dcnn_model = load_model(exper, simple=False)
    exper.optimizer = OPTIMIZER_DICT[exper.config.optimizer](dcnn_model.parameters(), lr=exper.run_args.lr)

    exper.batches_per_epoch = 2
    exper.logger.info("Size data-set {} // number of epochs {} // batch-size {} // batches/epoch {}".format(
        dataset.__len__(), exper.run_args.epochs, exper.run_args.batch_size, exper.batches_per_epoch))
    num_val_runs = 0
    for epoch_id in range(exper.run_args.epochs):
        exper.epoch_id += 1
        start_time = time.time()
        exper.logger.info("Start epoch {}".format(exper.epoch_id))
        for batch_id in range(exper.batches_per_epoch):
            new_batch = TwoDimBatchHandler(exper)
            new_batch.generate_batch_2d(dataset.images, dataset.labels)

            # print("new_batch.b_images", new_batch.b_images.shape)
            b_out = dcnn_model(new_batch.b_images)
            b_loss = dcnn_model.get_loss(b_out, new_batch.b_labels)
            # compute gradients w.r.t. model parameters
            b_loss.backward(retain_graph=False)
            exper.optimizer.step()
            # sum_grads = dcnn_model.sum_grads()
            exper.epoch_stats["mean_loss"][epoch_id] += b_loss.data.cpu().squeeze().numpy()
            dcnn_model.zero_grad()

        exper.epoch_stats["mean_loss"][epoch_id] *= 1./float(exper.batches_per_epoch)
        if exper.run_args.val_freq != 0 and (exper.epoch_id % exper.run_args.val_freq == 0 or
                                             exper.epoch_id == exper.run_args.epochs):
            # validate model
            num_val_runs += 1
            val_batch = TwoDimBatchHandler(exper)
            val_batch.generate_batch_2d(dataset.val_images, dataset.val_labels)
            dcnn_model.eval()
            b_out = dcnn_model(val_batch.b_images)
            val_loss = dcnn_model.get_loss(b_out, val_batch.b_labels)
            val_loss = val_loss.data.cpu().squeeze().numpy()[0]
            # store epochID and validation loss
            exper.val_stats["mean_loss"][num_val_runs-1] = np.array([exper.epoch_id, val_loss])
            exper.logger.info("Model validation in epoch {}: current loss {:.3f}".format(exper.epoch_id, val_loss))
            dcnn_model.train()

        if exper.run_args.chkpnt and (exper.epoch_id % 100 == 0 or exper.epoch_id == exper.run_args.epochs):
            save_checkpoint(exper, {'epoch': exper.epoch_id,
                                    'state_dict': dcnn_model.state_dict(),
                                    'best_prec1': 0.},
                            False, dcnn_model.__class__.__name__)
            # save exper statistics
            exper.save()
        end_time = time.time()
        total_time = end_time - start_time
        exper.logger.info("End epoch {}: mean loss: {:.3f} / duration {:.2f} seconds".format(
            exper.epoch_id,
            exper.epoch_stats["mean_loss"][
            epoch_id],
            total_time))
    del dataset
    del dcnn_model


def trainingv2(exper):
    traindir = os.path.join(config.data_dir, "train")
    filetype = "*image*.nii"
    images, labels, traincount = loadImageDir(glob.glob(traindir + os.path.sep + filetype), nclass=3)

    dcnn_model = load_model(exper)
    exper.optimizer = OPTIMIZER_DICT[exper.config.optimizer](dcnn_model.parameters(), lr=exper.run_args.lr)

    exper.batches_per_epoch = (len(images) / exper.run_args.batch_size)
    exper.logger.info("Size data-set {} // number of epochs {} // batch-size {} // batches/epoch {}".format(
        len(images), exper.run_args.epochs, exper.run_args.batch_size, exper.batches_per_epoch))

    for epoch_id in range(exper.run_args.epochs):
        start_time = time.time()
        exper.logger.info("Start epoch {}".format(epoch_id + 1))
        for batch_id in range(exper.batches_per_epoch):
            b_images, b_labels, _, _ = generateBatch2D(images, labels, nclass=3, nsamp=128, classcount=traincount)
            b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()))
            b_labels = Variable(torch.LongTensor(torch.from_numpy(b_labels.astype(int))))
            if exper.run_args.cuda:
                b_images = b_images.cuda()
                b_labels = b_labels.cuda()
            b_out = dcnn_model(b_images)
            b_loss = dcnn_model.get_loss(b_out, b_labels)
            # compute gradients w.r.t. model parameters
            b_loss.backward(retain_graph=False)
            exper.optimizer.step()
            sum_grads = dcnn_model.sum_grads()
            exper.logger.info("Current batch-loss {:.4f} / gradients {:.3f}".format(
                b_loss.data.cpu().squeeze().numpy()[0], sum_grads))
            dcnn_model.zero_grad()

        end_time = time.time()
        total_time = end_time - start_time
        exper.logger.info("End epoch {}: duration {:.2f} seconds".format(epoch_id + 1, total_time))

    del dcnn_model


def main():
    args = do_parse_args()

    exper = Experiment(config, run_args=args, set_seed=True)
    exper.start()
    print_flags(exper)

    if args.cmd == 'train':
        training(exper)
    elif args.cmd == "trainv2":
        trainingv2(exper)
    elif args.cmd == 'test':
        raise NotImplementedError("test mode is not yet implemented")


if __name__ == '__main__':
    main()
