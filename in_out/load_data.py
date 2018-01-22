import SimpleITK as sitk
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
# from matplotlib import cm

from torch.utils.data import Dataset
from utils.config import config


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    mri_scan = sitk.GetArrayFromImage(itkimage)

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


class LASunnyBrooksMRI(BaseImageDataSet):

    def __init__(self, data_dir, search_mask=None, transform=False, conf_obj=None, load_func="load_itk"):
        super(LASunnyBrooksMRI, self).__init__(data_dir, conf_obj)
        self.transform = transform
        self.search_mask = search_mask
        self.load_func = load_func
        self.crawl_directory()

    def crawl_directory(self):
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
                    mri_scan, origin, spacing = load_itk(fname)
                    self.images.append(mri_scan)
                    self.origins.append(origin)
                    self.spacings.append(spacing)
                    print("Info: Loading image {}".format(fname))
                    # construct the ground truth file filename
                    ext = fname[fname.find("."):]
                    ground_truth_img = os.path.join(abs_path, config.dflt_label_name + ext)
                    if os.path.exists(ground_truth_img):
                        gt_scan, _, _ = load_itk(ground_truth_img)
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


# dtaset = LASunnyBrooksMRI(data_dir="/home/jogi/git/repository/dcnn_mri_seg/data/", search_mask=config.dflt_image_name + ".mhd")
# dtaset.normalize_values(perc_high=95, axis=0)

# dtaset.save_to_numpy()


