import abc
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from in_out.load_data import write_numpy_to_image
from in_out.load_data import HVSMR2016CardiacMRI


def generate_rgb_image(org_img, classes=[1, 2]):
    """
    Assuming org_img is [classes, width, height, depth]
    we give each class a different color and reshape image so that classes is at the end



    """
    img_out = np.zeros(org_img.shape)

    colors = {1: np.array([61, 179, 11]),
              2: np.array([233, 48, 1])}
    r, g, b = org_img

    cls = classes[0]
    cls_Areas = (r == cls)
    img_out[0][cls_Areas] = colors[cls][0]
    img_out[1][cls_Areas] = colors[cls][1]
    img_out[2][cls_Areas] = colors[cls][2]

    cls = classes[1]
    cls_Areas = (r == cls)
    img_out[0][cls_Areas] = colors[cls][0]
    img_out[1][cls_Areas] = colors[cls][1]
    img_out[2][cls_Areas] = colors[cls][2]

    # we need to swap 3-rgb channel to last dim, and as usual swap dim 0 and 2 (1 with 3)
    img_out = np.reshape(img_out, (img_out.shape[3], img_out.shape[2], img_out.shape[1], img_out.shape[0]))

    return img_out


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

    def __init__(self, exper, is_train=True, batch_size=None, num_classes=3):
        if batch_size is None:
            self.batch_size = exper.run_args.batch_size
        else:
            self.batch_size = batch_size

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
        if self.b_labels_per_class is not None:
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

    def __init__(self, image, spacing=None, use_cuda=False, num_of_classes=3, batch_size=1, labels=None):
        """
            Input is an image passed as 3D numpy array  [x, y, z] axis
            Currently assuming that image is already scaled to isotropic size of 0.65 mm in all dimensions
            and that intensity values are normalized.

            We are currently assuming that the test-images are already a) normalized/scaled

        """
        self.b_images = []
        self.b_labels = []
        self.image = image
        self.labels = labels
        self.spacing = spacing
        self.view = None
        if labels is None:
            self.no_labels = True
        else:
            self.no_labels = False

        self.num_of_views = 0
        self.num_classes = num_of_classes
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.out_img = np.zeros((self.num_classes, self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        self.overlay_images = {'myocardium': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2])),
                               'bloodpool': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))}

    def __call__(self, model, exper_hdl):

        start_dt = time.time()
        print("INFO - Be patient - Generating overlays for {} view".format(self.view))
        np_pred = []
        for b, b_image in enumerate(self.b_images):
            b_image = Variable(torch.from_numpy(b_image).float(), volatile=True)
            if not self.no_labels:
                b_labels = Variable(torch.from_numpy(self.b_labels[b].astype(int)), volatile=True)
            if exper_hdl.exper.run_args.cuda:
                b_image = b_image.cuda()
                if not self.no_labels:
                    b_labels = b_labels.cuda()

            b_predictions = model(b_image)
            # print(b_predictions.size())
            np_pred.append(b_predictions.data.cpu().numpy())
            if not self.no_labels:
                val_loss = model.get_loss(b_predictions, b_labels)
                val_loss = val_loss.data.cpu().squeeze().numpy()[0]
                # compute dice score for both classes (myocardium and bloodpool)
                dice = HVSMR2016CardiacMRI.compute_accuracy(b_predictions, b_labels)
                exper_hdl.logger.info("Model testing: current loss {:.3f}\t "
                                      "dice-coeff(myo/blood) {:.3f}/{:.3f}".format(val_loss,
                                                                              dice[0], dice[1]))
        # because we average the predictions (for the three views) we need to keep track of the
        # views we generated
        self.num_of_views += 1
        np_image = np.concatenate(np_pred, axis=0)
        print("Image shape in/out ", self.image.shape, np_image.shape, self.view)
        for i in np.arange(np_image.shape[0]):
            # print("Image slices shapes ", (np_image[i, :, :, :]).shape)
            if self.view == "axial":
                self.out_img[:, :, :, i] += np_image[i, :, :, :]
            elif self.view == "saggital":
                self.out_img[:, i, :, :] += np_image[i, :, :, :]
            elif self.view == "coronal":
                self.out_img[:, :, i, :] += np_image[i, :, :, :]

        total_time = time.time() - start_dt
        print("INFO - Generating overlays took {:.3f} seconds ".format(total_time))

    def generate_overlays(self, exper_hdl):
        # attenuate noise
        if hasattr(exper_hdl.exper.config, 'noise_threshold'):
            self.out_img[self.out_img <= exper_hdl.exper.config.noise_threshold] = 0.
        else:
            self.out_img[self.out_img <= 0.01] = 0.
        # take the average of the "views" before we determine the class label for segmentation
        self.out_img *= 1./float(self.num_of_views)
        self.out_img = HVSMR2016CardiacMRI.get_pred_class_labels(self.out_img, axis=0)
        # Save the overlays for myocardium & bloodpool
        abs_out_filename = os.path.join(exper_hdl.exper.config.root_dir,
                                        os.path.join(exper_hdl.exper.output_dir,
                                                     exper_hdl.exper.config.figure_path))
        if not os.path.exists(abs_out_filename):
            print("Output path does not exist {}".format(abs_out_filename))
            abs_out_filename = os.environ.get('HOME')
            print("Using {} instead".format(abs_out_filename))
        myo_filename = os.path.join(abs_out_filename, "test_myocardium.nii")
        # average over the number of axis that we added to the final image
        myocardium_img = self.out_img[exper_hdl.exper.config.class_lbl_myocardium, :, :, :]
        write_numpy_to_image(myocardium_img, myo_filename, swap_axis=True, spacing=self.spacing)

        bloodpool_filename = os.path.join(abs_out_filename, "test_bloodpool.nii")
        # average over the number of axis that we added to the final image
        bloodpool_img = self.out_img[exper_hdl.exper.config.class_lbl_bloodpool, :, :, :]
        write_numpy_to_image(bloodpool_img, bloodpool_filename, swap_axis=True, spacing=self.spacing)

        # in order to generate the final image we sum over the first axis (contains class labels).
        full_img_filename = os.path.join(abs_out_filename, "test_full_img.nii")
        test_all_overlays = np.swapaxes(self.out_img, 1, 3)
        print("Full image out shape ", test_all_overlays.shape)
        write_numpy_to_image(test_all_overlays, full_img_filename, swap_axis=False, spacing=self.spacing)
        # test the deployment

    def generate_3D_batches(self, s_axis="axial"):
        """
        Assuming image is in [x, y, z] sequence

        """
        # important, we need to reset the batch lists...
        self.b_images = []
        self.b_labels = []
        self.view = s_axis
        shape_labels = None
        if s_axis == 'axial':
            # z-axis
            iter_axis = 2
            shape_images = tuple((self.batch_size, 1, self.image.shape[0] + (2 * HVSMR2016CardiacMRI.pad_size),
                                  self.image.shape[1] + (2 * HVSMR2016CardiacMRI.pad_size)))

            if not self.no_labels:
                shape_labels = tuple((self.batch_size, 1, self.labels.shape[0], self.labels.shape[1]))
            pad_image = np.pad(self.image, ((HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size),
                                       (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size), (0, 0)),
                               'constant', constant_values=(0,)).astype(HVSMR2016CardiacMRI.pixel_dta_type)
        elif s_axis == "saggital":
            # x-axis
            iter_axis = 0
            shape_images = tuple((self.batch_size, 1, self.image.shape[1] + (2 * HVSMR2016CardiacMRI.pad_size),
                                  self.image.shape[2] + (2 * HVSMR2016CardiacMRI.pad_size)))
            if not self.no_labels:
                shape_labels = tuple((self.batch_size, 1, self.labels.shape[1], self.labels.shape[2]))
            pad_image = np.pad(self.image, ((0, 0),
                                       (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size),
                                       (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size)),
                               'constant', constant_values=(0,)).astype(HVSMR2016CardiacMRI.pixel_dta_type)
        elif s_axis == "coronal":
            # y-axis
            iter_axis = 1
            shape_images = tuple((self.batch_size, 1, self.image.shape[0] + (2 * HVSMR2016CardiacMRI.pad_size),
                                  self.image.shape[2] + (2 * HVSMR2016CardiacMRI.pad_size)))
            if not self.no_labels:
                shape_labels = tuple((self.batch_size, 1, self.labels.shape[0], self.labels.shape[2]))
            pad_image = np.pad(self.image, ((HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size),
                                       (0, 0), (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size)),
                               'constant', constant_values=(0,)).astype(HVSMR2016CardiacMRI.pixel_dta_type)
        else:
            raise ValueError("Only axial/saggital/coronal view are supported for parameter s_axis "
                             "use passed {}".format(s_axis))

        def get_slice_borders(dim_size):

            chunks = dim_size / self.batch_size
            rest = dim_size % self.batch_size
            slices = [i * self.batch_size for i in np.arange(1, chunks + 1)]

            if rest != 0:
                slices.extend([slices[-1] + rest])

            return slices

        if self.image.shape[iter_axis] - self.batch_size < 0:
            # we can process all x-axis slices in one batch
            slices = [self.image.shape[iter_axis]]
            self.batch_size = self.image.shape[iter_axis]
        else:
            slices = get_slice_borders(self.image.shape[iter_axis])

        start_slice = 0
        # print(self.image.shape, pad_image.shape, shape_images, shape_labels)

        for end_slice in slices:
            if (end_slice - start_slice) != self.batch_size:
                shape_images = tuple((end_slice - start_slice, shape_images[1], shape_images[2], shape_images[3]))
                if not self.no_labels:
                    shape_labels = tuple((end_slice - start_slice, shape_labels[1], shape_labels[2], shape_labels[3]))
            b_images = np.zeros(shape_images)
            b_labels = np.zeros(shape_labels)
            # print("Start slice/end slice ", start_slice, end_slice)
            for i in np.arange(end_slice - start_slice):

                # print("begin/end ", start_slice+i, end_slice)
                if iter_axis == 0:
                    b_images[i, 0, :, :] = pad_image[start_slice+i, :, :]
                    if not self.no_labels:
                        b_labels[i, 0, :, :] = self.labels[start_slice+i, :, :]
                if iter_axis == 1:
                    b_images[i, 0, :, :] = pad_image[:, start_slice+i, :]
                    if not self.no_labels:
                        b_labels[i, 0, :, :] = self.labels[:, start_slice+i, :]
                if iter_axis == 2:
                    b_images[i, 0, :, :] = pad_image[:, :, start_slice+i]
                    if not self.no_labels:
                        b_labels[i, 0, :, :] = self.labels[:, :, start_slice+i]

            self.b_images.append(b_images)
            if not self.no_labels:
                self.b_labels.append(b_labels)
            start_slice = end_slice

        del b_images
        del b_labels

    def call(self, exper_hdl, model):
        """

        !!!!! CURRENTLY NOT IN USE !!!!!
        """

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
            if self.calculate_dice:
                batch_labels = self.labels[:, :, start_slice:end_slice]
                batch_labels = np.reshape(batch_labels, (batch_labels.shape[2], batch_labels.shape[0],
                                                         batch_labels.shape[1]))
                dice = HVSMR2016CardiacMRI.compute_accuracy(pred_score, batch_labels)
                print(pred_score.size(), batch_labels.shape)
                print("Dice scores {:.3f} / {:.3f}".format(dice[0], dice[1]))
            pred_score = pred_score.data.cpu().numpy()
            # pred_score tensor is [batch_size, num_classes, x-axis, y-axis] hence we need to reshape to
            pred_score = np.reshape(pred_score, (pred_score.shape[1], pred_score.shape[2],
                                                 pred_score.shape[3], pred_score.shape[0]))
            out_img[:, :, :, start_slice:end_slice] = pred_score
            start_slice = end_slice

        del batch
        del aximage
        exit(1)
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
        if hasattr(exper_hdl.exper.config, 'noise_threshold'):
            out_img[out_img <= exper_hdl.exper.config.noise_threshold] = 0.
        else:
            out_img[out_img <= 0.01] = 0.
        #
        # sharp_overlays = HVSMR2016CardiacMRI.get_pred_class_labels(out_img & 1./3.)
        # Save the overlays for myocardium & bloodpool
        abs_out_filename = os.path.join(exper_hdl.exper.config.root_dir,
                                        os.path.join(exper_hdl.exper.output_dir, exper_hdl.exper.config.figure_path))
        if not os.path.exists(abs_out_filename):
            print("Output path does not exist {}".format(abs_out_filename))
            abs_out_filename = os.environ.get('HOME')
            print("Using {} instead".format(abs_out_filename))
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
        # test the deployment
