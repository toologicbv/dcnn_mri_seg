"""
    Surface-to-surface distance measure
    Reference: https://mlnotebook.github.io/post/surface-distance-function/
"""

import numpy as np
from scipy.ndimage import morphology


def surfd(input1, input2, sampling=1, connectivity=1):
    """
    input1 -       the segmentation that has been created. It can be a multi-class segmentation,
                   but this function will make the image binary.
    input2 -       the GT segmentation against which we wish to compare input1
    sampling -     the pixel resolution or pixel size. This is entered as an n-vector where n is equal to the
                   number of dimensions in the segmentation i.e. 2D or 3D.
                   The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
    connectivity - creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around
                   which the function looks for neighbouring pixels. Typically, this is defined as
                   a six-neighbour kernel which is the default behaviour of this function.
    """

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    """
        What can you do with the returned result?
        (1) Mean Surface Distance
        msd = surface_distance.mean()
        (2) Residual Mean Square Distance
        rms = np.sqrt((surface_distance**2).mean())
        (3) Hausdorff Distance
        hd  = surface_distance.max()
    """

    return sds
