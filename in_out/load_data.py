import SimpleITK as sitk
import numpy as np
import os
import glob
# import h5py
import gc

import matplotlib.pyplot as plt
# from matplotlib import cm

from torch.utils.data import Dataset
from utils.config import config


def load_mhd_to_numpy(filename, data_type="float32"):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    mri_scan = sitk.GetArrayFromImage(itkimage).astype(data_type)

    # Read the origin of the mri_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return mri_scan, origin, spacing


def crawl_dir(in_dir, load_func="load_itk", pattern="*.mhd", logger=None):
    """
    Searches for files that match the pattern-parameter and assumes that there also
    exist a <filename>.raw for this mhd file
    :param in_dir:
    :param load_func:
    :param pattern:
    :param logger:
    :return: python list with
    """
    im_arrays = []
    gt_im_arrays = []

    pattern = os.path.join(in_dir, pattern)
    for fname in glob.glob(pattern):
        mri_scan, origin, spacing = load_func(fname)
        logger.info("Loading {}".format(fname, ))
        im_arrays.append((mri_scan, origin, spacing))

    return im_arrays, gt_im_arrays


def rescale_image(img, perc_low=5, perc_high=95, axis=None):
    # flatten 3D image to 1D and determine percentiles for recaling
    lower, upper = np.percentile(img, [perc_low, perc_high], axis=axis)
    # set new normalized image
    img = (img - lower) * 1. / (upper - lower)
    return img


def normalize_image(img, axis=None):
    img = (img - np.mean(img, axis=axis) / np.std(img, axis=axis))
    return img


class BaseImageDataSet(Dataset):
    def __init__(self, data_dir, conf_obj=None):
        if data_dir is None:
            if not conf_obj is None:
                data_dir = config.data_dir
            else:
                raise ValueError("parameter {} is None, cannot load images".format(data_dir))
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self.search_mask = None
        self.images = None
        self.images_raw = None
        self.labels = None
        self.origins = None
        self.spacings = None

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.images)

    def crawl_directory(self):
        pass

    def save_to_numpy(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = "img_np"
        if abs_path is None:
            abs_path = config.data_dir
        if not os.path.exists(abs_path):
            raise IOError("{} does not exist".format(abs_path))

        for idx in np.arange(self.__len__()):
            filename = os.path.join(abs_path, file_prefix + str(idx+1) + ".npz")
            image = self.images[idx]
            label = self.labels[idx]
            try:
                np.savez(filename, image=image, label=label)
            except IOError:
                raise IOError("Can't save {}".format(filename))


class LASunnyBrooksMRI(BaseImageDataSet):

    def __init__(self, data_dir, search_mask=None, transform=False, conf_obj=None, load_func="load_mhd_to_numpy"):
        super(LASunnyBrooksMRI, self).__init__(data_dir, conf_obj)
        self.transform = transform
        self.search_mask = search_mask
        self.load_func = load_func
        self.crawl_directory()
        self.images = []
        self.labels = []

    def load_images_from_dir(self):
        """
            Searches for files that match the pattern-parameter and assumes that there also
            exist a <filename>.raw for this mhd file

            """
        self.images = []
        self.images_raw = []
        self.labels = []
        self.origins = []
        self.spacings = []
        dirs = os.listdir(self.data_dir)
        dirs.sort()
        for file_name in dirs:
            abs_path = os.path.join(self.data_dir, file_name)

            if os.path.isdir(os.path.dirname(abs_path)):

                search_mask = os.path.join(abs_path, self.search_mask)
                for fname in glob.glob(search_mask):
                    mri_scan, origin, spacing = self.load_func(fname)
                    self.images.append(mri_scan)
                    self.origins.append(origin)
                    self.spacings.append(spacing)
                    print("Info: Loading image {}".format(fname))
                    # construct the ground truth file filename
                    ext = fname[fname.find("."):]
                    ground_truth_img = os.path.join(abs_path, config.dflt_label_name + ext)
                    if os.path.exists(ground_truth_img):
                        gt_scan, _, _ = self.load_func(ground_truth_img)
                        self.labels.append(gt_scan)
                    else:
                        raise IOError("{} does not exist".format(ground_truth_img))
                    print("gt {}".format(ground_truth_img))

    def __getitem__(self, index):
        assert index <= self.__len__()
        return tuple((self.images[index], self.labels[index]))

    def __getitem_img__(self, index):
        assert index <= self.__len__()
        return self.images[index]

    def __getitem_label__(self, index):
        assert index <= self.__len__()
        return self.labels[index]

    def __get_img_flattened__(self, raw_data=False):

        all_data = np.empty(0)
        l_data_set = self.__len__()

        for idx in np.arange(l_data_set):
            if raw_data:
                img = self.images_raw[idx]
            else:
                img = self.images[idx]
            all_data = np.append(all_data, np.ravel(img))

        return all_data

    def normalize_values(self, perc_low=0, perc_high=95, axis=0):

        self.images_raw = []
        for idx in np.arange(self.__len__()):
            img = self.images[idx]
            # save un-normalized image for comparison
            self.images_raw.append(np.copy(img))
            lower, upper = np.percentile(np.ravel(img), [perc_low, perc_high], axis=axis)
            # set new normalized image
            img = (img - lower) * 1./(upper-lower)
            self.images[idx] = img

    def save_to_numpy(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = "img_np"
        if abs_path is None:
            abs_path = config.data_dir
        if not os.path.exists(abs_path):
            raise IOError("{} does not exist".format(abs_path))

        for idx in np.arange(self.__len__()):
            filename = os.path.join(abs_path, file_prefix + str(idx+1) + ".npz")
            image = self.images[idx]
            label = self.labels[idx]
            try:
                np.savez(filename, image=image, label=label)
            except IOError:
                raise IOError("Can't save {}".format(filename))

    def show_histogram(self, raw=False, bins=50):

        all_data = self.__get_img_flattened__()
        if raw:
            plots = 2
        else:
            plots = 1
        plt.figure(figsize=(14, 7))
        plt.subplot(1, plots, 1)
        plt.hist(all_data, bins=bins)
        if raw:
            all_data_raw = self.__get_img_flattened__(raw_data=raw)
            plt.subplot(1, plots, 2)
            plt.hist(all_data_raw, bins=bins)
            del all_data_raw
        plt.show()
        plt.close()
        del all_data


class HVSMR2016CardiacMRI(BaseImageDataSet):

    pixel_dta_type = 'float32'
    pad_size = 65

    def __init__(self, data_dir, search_mask=None, nclass=3, transform=False, conf_obj=None,
                 load_func=load_mhd_to_numpy, norm_scale=None, mode="train", load_type="raw"):
        """
        The images are already resampled to an isotropic 3D size of 0.65mm x 0.65 x 0.65


        :param data_dir: root directory
        :param search_mask:
        :param nclass:
        :param transform:
        :param conf_obj:
        :param load_func: currently only load_mhd_to_numpy is supported
        :param norm_scale: takes arguments "normalize" or "rescale"
        :param mode: takes "train", "test", "valid"
        """
        super(HVSMR2016CardiacMRI, self).__init__(data_dir, conf_obj)
        self.transform = transform
        self.norm_scale = norm_scale
        self.search_mask = search_mask
        self.load_func = load_func
        self.load_type = load_type
        self.mode = mode
        # Note, this list will contain len() image slices...2D!
        self.images = []
        self.labels = []
        self.origins = []
        self.spacings = []
        self.no_class = nclass
        self.class_count = np.zeros(self.no_class)
        if self.load_type == "raw":
            self.load_images_from_dir()
        elif self.load_type == "numpy":
            self.load_numpy_arr_from_dir()
        else:
            raise ValueError("Load mode {} is not supported".format(self.load_type))

    def __getitem__(self, index):
        assert index <= self.__len__()
        return tuple((self.images[index], self.labels[index]))

    def _get_file_lists(self):

        img_file_list = []
        label_file_list = []

        if self.mode == "train":
            input_dir = os.path.join(self.data_dir, "train")
        elif self.mode == "test":
            input_dir = os.path.join(self.data_dir, "test")
        else:
            raise ValueError("Loading mode {} is currently not supported (train/test)".format(self.mode))

        search_mask = os.path.join(input_dir, self.search_mask)

        for fname in glob.glob(search_mask):
            img_file_list.append(fname)
            label_file_list.append(fname.replace("image", "label"))

        return img_file_list, label_file_list

    def load_images_from_dir(self):
        """
            Searches for files that match the search_mask-parameter and assumes that there are also
            reference aka label images accompanied with each image

        """
        files_loaded = 0

        img_file_list, label_file_list = self._get_file_lists()
        for i, file_name in enumerate(img_file_list):
            print("Loading image+label from {}".format(file_name))
            mri_scan, origin, spacing = self.load_func(file_name, data_type=HVSMR2016CardiacMRI.pixel_dta_type)
            if self.norm_scale == "normalize":
                mri_scan = normalize_image(mri_scan, axis=None)
            elif self.norm_scale == "rescale":
                mri_scan = rescale_image(mri_scan, axis=None)
            else:
                # no rescaling or normalization
                print("Info - No rescaling or normalization applied to image!")
            # add a front axis to the numpy array, will use that to concatenate the image slices
            self.origins.append(origin)
            self.spacings.append(spacing)
            print("{} / {}".format(file_name,  label_file_list[i]))
            label, _, _ = self.load_func(label_file_list[i])
            # augment the image with additional rotated slices
            self._augment_data(mri_scan, label, pad_size=HVSMR2016CardiacMRI.pad_size)
            for class_label in range(self.no_class):
                self.class_count[class_label] += np.sum(label == class_label)
            files_loaded += 1

    def _augment_data(self, image, label, pad_size=0):
        """
        Adds all original and rotated image slices to self.images and self.labels objects
        :param image:
        :param label:
        :param pad_size:
        :return:
        """

        def rotate_slice(image_slice, label_slice):
            # PAD IMAGE
            for rots in range(4):
                section = np.pad(image_slice, pad_size, 'constant', constant_values=(0,)).astype(
                    HVSMR2016CardiacMRI.pixel_dta_type)
                self.images.append(section)
                self.labels.append(label_slice)
                # rotate for next iteration
                image_slice = np.rot90(image_slice)
                label_slice = np.rot90(label_slice)

        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(image.shape[2]):
            label_slice = label[:, :, z]
            image_slice = image[:, :, z]
            rotate_slice(image_slice, label_slice)

        for y in range(image.shape[1]):
            label_slice = np.squeeze(label[:, y, :])
            image_slice = np.squeeze(image[:, y, :])
            rotate_slice(image_slice, label_slice)

        for x in range(image.shape[0]):
            label_slice = np.squeeze(label[x, :, :])
            image_slice = np.squeeze(image[x, :, :])
            rotate_slice(image_slice, label_slice)

    def load_numpy_arr_from_dir(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = config.numpy_save_filename

        if abs_path is None:
            abs_path = self.data_dir

        if self.mode == "train":
            out_dir = os.path.join(abs_path, "train")
        else:
            out_dir = os.path.join(abs_path, "test")

        search_mask = os.path.join(out_dir, file_prefix + "*.npz")
        print("Info - Looking for files with search_mask {}".format(search_mask))
        for fname in glob.glob(search_mask):
            print("Info - Loading numpy objects from {}".format(fname))
            numpy_ar = np.load(fname)
            self.images.extend(list(numpy_ar["images"]))
            self.labels.extend(list(numpy_ar["labels"]))
            if self.class_count is None:
                self.class_count = numpy_ar["class_count"]

    def save_to_numpy(self, file_prefix=None, abs_path=None):

        if file_prefix is None:
            file_prefix = config.numpy_save_filename

        if abs_path is None:
            abs_path = self.data_dir

        if self.mode == "train":
            out_filename = os.path.join(abs_path, "train")
        else:
            out_filename = os.path.join(abs_path, "test")

        try:
            print("Trying to save data to directory {}".format(abs_path))

            chunksize = 1000
            start = 0
            end = chunksize
            for c, chunk in enumerate(np.arange(chunksize)):
                filename = os.path.join(out_filename, file_prefix + str(chunk) + ".npz")
                np.savez(filename, images=self.images[start:end],
                         labels=self.labels[start:end],
                         class_count=self.class_count)
                filename_hdf5 = os.path.join(out_filename, file_prefix + str(chunk) + ".hd5f")
                # h5f = h5py.File(filename_hdf5, 'w')
                # h5f.create_dataset('images', data=self.images[start:end])
                # h5f.create_dataset('labels', data=self.labels[start:end])
                # h5f.create_dataset('class_count', data=self.class_count)
                # h5f.close()
                start += chunksize
                end += chunksize
                if c == 3:
                    break
        except IOError:
            raise IOError("Can't save {}".format(filename))


# dataset = HVSMR2016CardiacMRI(data_dir=config.data_dir, search_mask=config.dflt_image_name + ".nii",
#                              norm_scale="rescale", load_type="numpy")
# print("Length data set {}/{}".format(dataset.__len__(), len(dataset.labels)))
# dataset.save_to_numpy()
# del dataset

