import abc
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from in_out.load_data import write_numpy_to_image
from in_out.load_data import HVSMR2016CardiacMRI


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
    pixel_dta_type = "float32"

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
        self.config = exper.config

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels = self.b_labels.cuda()
        self.b_labels_per_class = self.b_labels_per_class.cuda()

    def __call__(self, exper, network):
        print("Batch size {}".format(exper.run_args.batch_size))

    def backward(self, *args):
        pass

    def generate_batch_2d(self, images, labels, save_batch=False):
        b_images = np.zeros((self.batch_size, 1, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes, self.patch_size + 1, self.patch_size + 1))
        b_labels = np.zeros((self.batch_size, 1, self.patch_size + 1, self.patch_size + 1))
        num_images = len(images)
        # img_nums = []
        for idx in range(self.batch_size):
            ind = np.random.randint(0, num_images)
            # img_nums.append(str(ind))
            img = images[ind]
            label = labels[ind]

            offx = np.random.randint(0, img.shape[0] - self.ps_wp)
            offy = np.random.randint(0, img.shape[1] - self.ps_wp)

            img = img[offx:offx + self.ps_wp, offy:offy + self.ps_wp]

            b_images[idx, 0, :, :] = img
            label = label[offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]

            b_labels[idx, 0, :, :] = label

            for cls in range(self.num_classes):
                b_labels_per_class[idx, cls, :, :] = (label == cls).astype('int16')
        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()))
        self.b_labels = Variable(torch.LongTensor(torch.from_numpy(b_labels.astype(int))))
        self.b_labels_per_class = Variable(torch.LongTensor(torch.from_numpy(b_labels_per_class.astype(int))))
        if save_batch:
            self.save_batch_img_to_files()

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels
        del b_labels_per_class

    def save_batch_img_to_files(self):
        for i in np.arange(self.batch_size):
            filename_img = os.path.join(self.config.data_dir, "b_img" + str(i+1).zfill(2) + ".nii")
            write_numpy_to_image(self.b_images[i].data.cpu().numpy(), filename=filename_img)

            lbl = self.b_labels[i].data.cpu().numpy()
            print(np.unique(lbl))
            for l in np.unique(lbl):
                if l != 0:
                    cls_lbl = self.b_labels_per_class[i, l].data.cpu().numpy()
                    filename_lbl = os.path.join(self.config.data_dir, "b_lbl" + str(i + 1).zfill(2) + "_" + str(l) + ".nii")
                    lbl = np.pad(cls_lbl, 65, 'constant', constant_values=(0,)).astype("float32")
                    write_numpy_to_image(lbl, filename=filename_lbl)


class TestHandler(object):

    def __init__(self, image, use_cuda=False, num_of_classes=3, batch_size=1):
        """
            Input is an image passed as 3D numpy array  [x, y, z] axis
            Currently assuming that image is already scaled to isotropic size of 0.65 mm in all dimensions
            and that intensity values are normalized.

            We are currently assuming that the test-images are already a) normalized/scaled

        """
        self.image = image
        self.use_cuda = use_cuda
        self.num_of_classes = num_of_classes
        self.batch_size = batch_size
        self.overlay_images = {}

        if self.use_cuda:
            self.cuda()

    def cuda(self):
        pass

    def __call__(self, exper_hdl, model):

        def get_slice_borders(dim_size):

            chunks = dim_size / self.batch_size
            rest = dim_size % self.batch_size
            slices = [i * self.batch_size for i in np.arange(1, chunks + 1)]

            if rest != 0:
                slices.extend([slices[-1] + rest])

            return slices

        # first set model into eval-mode to save memory
        model.eval()
        # declare the final output image
        print("INFO - Generating overlays for image with shape {}".format(str(self.image.shape)))
        print("INFO - Start with generating slices for z-axis")
        start_dt = time.time()
        out_img = np.zeros((self.num_of_classes, self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        """
            Start with the axial dimension (z-axis)
        """
        dim_size = self.image.shape[2]
        slices = get_slice_borders(dim_size)
        aximage = np.pad(self.image,
                         ((exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size),
                          (exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size),
                          (0, 0)), 'constant', constant_values=(0,)).astype('float32')
        start_slice = 0
        for end_slice in slices:
            batch = aximage[:, :, start_slice:end_slice]
            batch = np.reshape(batch, (batch.shape[2], batch.shape[0], batch.shape[1]))
            batch = np.expand_dims(batch, axis=1)
            batch = Variable(torch.from_numpy(batch).float(), volatile=True)
            if self.use_cuda:
                batch = batch.cuda()
            pred_score = model(batch)
            pred_score = pred_score.data.cpu().numpy()
            # pred_score tensor is [batch_size, num_classes, x-axis, y-axis] hence we need to reshape to
            pred_score = np.reshape(pred_score, (pred_score.shape[1], pred_score.shape[2],
                                                 pred_score.shape[3], pred_score.shape[0]))
            out_img[:, :, :, start_slice:end_slice] = pred_score
            start_slice = end_slice

        del batch
        del aximage
        """
            Followed by saggital dimension (x-axis)
        """
        print("INFO - Generating slices for x-axis")
        dim_size = self.image.shape[0]
        slices = get_slice_borders(dim_size)
        sagimage = np.pad(self.image, ((0, 0),
                                       (exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size),
                                       (exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size)),
                                        'constant', constant_values=(0,)).astype('float32')
        start_slice = 0
        for end_slice in slices:
            batch = sagimage[start_slice:end_slice, :, :]
            batch = np.expand_dims(batch, axis=1)
            batch = Variable(torch.from_numpy(batch).float(), volatile=True)
            if self.use_cuda:
                batch = batch.cuda()
            pred_score = model(batch)
            pred_score = pred_score.data.cpu().numpy()
            # pred_score tensor is [x-axis, num_classes, y-axis, z-axis] hence we need to reshape to
            pred_score = np.reshape(pred_score, (pred_score.shape[1], pred_score.shape[0],
                                                 pred_score.shape[2], pred_score.shape[3]))
            out_img[:, start_slice:end_slice, :, :] += pred_score
            start_slice = end_slice

        """
             Followed by coronal dimension (y-axis)
        """
        print("INFO - Generating slices for y-axis")
        start_slice = 0
        dim_size = self.image.shape[1]
        corimage = np.pad(self.image, ((exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size), (0, 0),
                                       (exper_hdl.exper.config.pad_size, exper_hdl.exper.config.pad_size)),
                          'constant', constant_values=(0,)).astype('float32')
        slices = get_slice_borders(dim_size)
        for end_slice in slices:
            batch = corimage[:, start_slice:end_slice, :]
            batch = np.reshape(batch, (batch.shape[1], batch.shape[0], batch.shape[2]))
            batch = np.expand_dims(batch, axis=1)
            batch = Variable(torch.from_numpy(batch).float(), volatile=True)
            if self.use_cuda:
                batch = batch.cuda()
            pred_score = model(batch)
            pred_score = pred_score.data.cpu().numpy()
            # pred_score tensor is [y-axis, num_classes, x-axis, z-axis] hence we need to reshape to
            pred_score = np.reshape(pred_score, (pred_score.shape[1], pred_score.shape[2],
                                                 pred_score.shape[0], pred_score.shape[3]))
            out_img[:, :, start_slice:end_slice, :] += pred_score
            start_slice = end_slice
        # set model back into training mode
        model.train()
        del batch
        del sagimage
        # attenuate noise
        out_img[out_img <= exper_hdl.exper.config.noise_threshold] = 0.
        #
        # sharp_overlays = HVSMR2016CardiacMRI.get_pred_class_labels(out_img & 1./3.)
        # Save the overlays for myocardium & bloodpool
        abs_out_filename = os.path.join(exper_hdl.exper.output_dir, exper_hdl.exper.config.figure_path)
        myo_filename = os.path.join(abs_out_filename, "test_myocardium.nii")
        # average over the number of axis that we added to the final image
        myocardium_img = out_img[exper_hdl.exper.config.class_lbl_myocardium, :, :, :] * 1./3.
        write_numpy_to_image(myocardium_img, myo_filename, swap_axis=True)

        bloodpool_filename = os.path.join(abs_out_filename, "test_bloodpool.nii")
        # average over the number of axis that we added to the final image
        bloodpool_img = out_img[exper_hdl.exper.config.class_lbl_bloodpool, :, :, :] * 1. / 3.
        write_numpy_to_image(bloodpool_img, bloodpool_filename, swap_axis=True)
        total_time = time.time() - start_dt
        print("INFO - Generating overlays took {:.3f} seconds ".format(total_time))
