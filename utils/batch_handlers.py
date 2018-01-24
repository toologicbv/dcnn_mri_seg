import abc

import numpy as np
import torch
from torch.autograd import Variable


class BatchHandler(object):
    __metaclass__ = abc.ABCMeta

    # static class variable to count the batches
    id = 0

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *args):
        pass


class TwoDimBatchHandler(BatchHandler):

    # we use a zero-padding of 65 on both dimensions, equals 130 positions
    patch_size_with_padding = 201
    patch_size = 70

    def __init__(self, exper, is_train=True, num_classes=3):
        self.batch_size = exper.run_args.batch_size
        self.is_train = is_train
        self.num_classes = num_classes
        self.ps_wp = TwoDimBatchHandler.patch_size_with_padding
        self.patch_size = TwoDimBatchHandler.patch_size
        self.is_cuda = exper.run_args.cuda

        # batch image patch
        self.b_images = None
        # batch reference image for the different classes (so for each reference class 1 image)
        self.b_labels = None
        # this objects holds for each image-slice the separate class labels, so one set for each class
        self.b_labels_per_class = None

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels = self.b_labels.cuda()
        self.b_labels_per_class = self.b_labels_per_class.cuda()

    def __call__(self, exper, network):
        print("Batch size {}".format(exper.run_args.batch_size))

    def backward(self, *args):
        pass

    def _generate_batch_2d(self, images, labels):
        b_images = np.zeros((self.batch_size, 1, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes, self.patch_size + 1, self.patch_size + 1))
        b_labels = np.zeros((1, self.batch_size, self.patch_size + 1, self.patch_size + 1))
        num_images = len(images)

        for idx in range(self.batch_size):
            ind = np.random.randint(0, num_images, 1)[0]
            img = images[ind]
            label = labels[ind]

            offx = np.random.randint(0, img.shape[0] - self.ps_wp)
            offy = np.random.randint(0, img.shape[1] - self.ps_wp)

            img = img[offx:offx + self.ps_wp, offy:offy + self.ps_wp]
            b_images[idx, 0, :, :] = img
            label = label[offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]

            b_labels[0, idx, :, :] = label

            for cls in range(self.num_classes):
                b_labels_per_class[idx, cls, :, :] = (label == cls).astype('int16')

        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()))
        self.b_labels = Variable(torch.FloatTensor(torch.from_numpy(b_labels).float()))
        self.b_labels_per_class = Variable(torch.FloatTensor(torch.from_numpy(b_labels_per_class).float()))

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels
        del b_labels_per_class

