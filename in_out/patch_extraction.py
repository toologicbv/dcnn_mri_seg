from scipy.ndimage.interpolation import map_coordinates, rotate
import numpy as np


def compute_crop(coordinate, width, axis_length):
    coordinate_from = coordinate - (width // 2)
    coordinate_to = coordinate + int(np.ceil(width / 2.0))

    padding_before = 0 if coordinate_from >= 0 else abs(coordinate_from)
    padding_after = 0 if coordinate_to < axis_length else coordinate_to - axis_length

    if padding_before > 0:
        coordinate_from = 0

    return coordinate_from, coordinate_to, padding_before, padding_after


def as_tuple(x, n, t=None, return_none=False):
    """Converts a value x into a tuple and ensures that the length is n"""
    if return_none and x is None:
        return None

    try:
        y = tuple(x)
        if len(y) != n:
            raise ValueError('Expected a single value or an iterable with length {0}, got {1} instead'.format(n, x))
    except TypeError:
        y = (x,) * n

    if t:
        y = tuple(t(v) for v in y)

    return y


def as_list(x, n, t=None, return_none=False):
    """
    Converts a value x into a list of length n.
    Unlike as_tuple(), a sequence with != n elements will be repeated n times instead of causing an exception!
    """
    if return_none and x is None:
        return None

    try:
        if len(x) == n:
            y = list(x)
        else:
            y = [x] * n
    except TypeError:
        y = [x] * n

    if t:
        y = [t(v) for v in y]

    return y


class PatchExtractor3D:
    """
    Extracts 2D, 2.5D and 3D patches from a 3D volume

    Parameters
    ----------
    image : numpy array
        3D image volume
        Comment Jorg (jan-2018):
        Makes the assumption that the input image has axial axis ordering: (x, y, z)

    voxel_spacing : iterable of 3 floats or None
        Spacing between voxels in the image

    pad_value : numeric
        Value used to pad extracted patches if they extent beyond the image boundaries

    spline_order : int
        Order of the spline used for interpolation (if necessary)

    dtype : numpy dtype
        Datatype of the extracted patches, theano.config.floatX or 'float32' is usually a good value.
        Defaults to the dtype of the image.
    """
    sagittal = 0  #: Value for axis parameter to extract sagittal patches/slices
    coronal = 1  #: Value for axis parameter to extract coronal patches/slices
    axial = 2  #: Value for axis parameter to extract axial patches/slices

    def __init__(self, image, voxel_spacing=None, pad_value=0, spline_order=1, dtype=None):
        self.image = np.asarray(image)
        if self.image.ndim != 3:
            raise ValueError('Expected a 3D volume, got array with {} dimensions instead'.format(self.image.ndim))

        self.voxel_spacing = as_tuple(voxel_spacing, 3, float, return_none=True)
        self.pad_value = pad_value
        self.spline_order = int(spline_order)

        if dtype is None:
            self.dtype = self.image.dtype
        else:
            self.dtype = np.dtype(dtype)

        # Store orthogonal views
        self.views = (
            np.transpose(self.image, axes=(1, 2, 0)),  # sagittal (y, z, x)
            np.transpose(self.image, axes=(0, 2, 1)),  # coronal  (x, z, y)
            self.image,  # axial    (x, y, z)
        )

    def interpolate(self, image, image_spacing, patch_center, patch_shape, patch_extent):
        patch_spacing = [e / s for e, s in zip(patch_extent, patch_shape)]

        sample_radius = [(ps / cs) * (s / 2.0) for ps, cs, s in zip(patch_spacing, image_spacing, patch_shape)]
        sample_points = [np.linspace(pc - sr, pc + sr, s) for pc, sr, s in zip(patch_center, sample_radius, patch_shape)]

        chunk_from = [max(0, int(np.floor(sp[0])) - 1) for sp in sample_points]
        chunk_to = [int(np.ceil(sp[-1])) + 1 for sp in sample_points]
        chunk = image[[slice(cf, ct) for cf, ct in zip(chunk_from, chunk_to)]]

        sample_grid = np.meshgrid(*[sp - cf for sp, cf in zip(sample_points, chunk_from)])
        return map_coordinates(chunk, sample_grid, order=self.spline_order, output=self.dtype, cval=self.pad_value)

    def extract_slice(self, i, axis=2):
        """Returns an entire slice (sagittal, coronal or axial view depending on the specified axis)"""
        s = self.views[axis][:, :, i]
        if axis != 2:
            s = np.fliplr(s)
        return s

    def extract_rect(self, center_voxel, shape, extent=None, axis=2, rotation_angle=None):
        """Extracts a 2D rectangular patch"""
        shape = as_tuple(shape, 2, int)
        extent = as_tuple(extent, 2, float, return_none=True)

        if rotation_angle is not None:
            # Extract a larger patch, rotate and crop
            radius = int(np.ceil(np.linalg.norm(shape, ord=2)))

            if extent:
                radius += radius % 2  # not sure why this is necessary, but without it the patch is slightly shifted
                patch_spacing = [e / s for e, s in zip(extent, shape)]
                extent = tuple(radius * ps for ps in patch_spacing)

            patch_with_padding = self.extract_rect(center_voxel, radius, extent, axis)
            rotated_patch = rotate(patch_with_padding, angle=rotation_angle, reshape=False, order=self.spline_order,
                                   output=self.dtype, cval=self.pad_value)

            patch_from = [(rps - s) // 2 for rps, s in zip(rotated_patch.shape, shape)]
            patch_to = [pf + s for pf, s in zip(patch_from, shape)]
            return rotated_patch[[slice(pf, pt) for pf, pt in zip(patch_from, patch_to)]]

        # Extract 2D slab orthogonal to the specified axis
        slab = self.views[axis][:, :, center_voxel[axis]]

        if axis == 0:
            i, j = 1, 2
        elif axis == 1:
            i, j = 0, 2
        else:
            i, j = 0, 1

        if extent:
            # Extract patch with interpolation
            if self.voxel_spacing is None:
                raise ValueError('Cannot perform interpolation if the voxel spacing of the image is unknown')

            slab_spacing = (self.voxel_spacing[i], self.voxel_spacing[j])
            patch_center = (center_voxel[i], center_voxel[j])
            patch = self.interpolate(slab, slab_spacing, patch_center, shape, extent).T
        else:
            # Extract patch without interpolation
            i_from, i_to, i_padding_before, i_padding_after = compute_crop(center_voxel[i], shape[0], slab.shape[0])
            j_from, j_to, j_padding_before, j_padding_after = compute_crop(center_voxel[j], shape[1], slab.shape[1])

            # Generate array with the proper size by padding with pad_value
            chunk = slab[i_from:i_to, j_from:j_to]
            paddings = ((i_padding_before, i_padding_after), (j_padding_before, j_padding_after))
            patch = np.pad(chunk, paddings, mode='constant', constant_values=self.pad_value).astype(self.dtype)

        # Make sure the patch orientation is correct
        if axis != 2:
            patch = np.fliplr(patch)  # Flip axis=1

        return patch

    def extract_ortho_rects(self, center_voxel, shape, extent=None, stack_axis=0, rotation_angle=None):
        """Extracts a set of three orthogonal 2D patches (sagittal, coronal, axial)"""
        extent = as_list(extent, 3)
        angles = as_tuple(rotation_angle, 3, float, return_none=True)
        return np.stack(
            [self.extract_rect(center_voxel, shape, extent=extent[i], axis=i, rotation_angle=angles[i] if angles else None) for i
             in range(3)],
            axis=stack_axis
        )

    def extract_cuboid(self, center_voxel, shape, extent=None):
        """Extracts a 3D patch"""
        shape = as_tuple(shape, 3, int)
        extent = as_tuple(extent, 3, float, return_none=True)

        if extent:
            # Extract patch with interpolation
            if self.voxel_spacing is None:
                raise ValueError('Cannot perform interpolation if the voxel spacing of the image is unknown')

            patch = self.interpolate(self.image, self.voxel_spacing, center_voxel, shape, extent)
            patch = np.flipud(np.rot90(patch))
        else:
            # Extract patch without interpolation
            crops = [compute_crop(cv, s, al) for cv, s, al in zip(center_voxel, shape, self.image.shape)]
            chunk = self.image[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
            paddings = tuple(c[2:4] for c in crops)
            patch = np.pad(chunk, paddings, mode='constant', constant_values=self.pad_value).astype(self.dtype)

        return patch


class PatchExtractor2D:
    """
    Extracts 2D patches from a 2D image

    Parameters
    ----------
    image : numpy array
        2D image
        Comment Jorg (jan-2018):
        Assumes input image has axial axis ordering (x, y).
        Init method will extend image with third axis assuming this is the z-axis.

    pixel_spacing : iterable of 2 floats or None
        Spacing between pixels in the image

    pad_value : numeric
        Value used to pad extracted patches if they extent beyond the image boundaries

    spline_order : int
        Order of the spline used for interpolation (if necessary)

    dtype : numpy dtype
        Datatype of the extracted patches, theano.config.floatX or 'float32' is usually a good value.
        Defaults to the dtype of the image.
    """
    def __init__(self, image, pixel_spacing=None, pad_value=0, spline_order=1, dtype=None):
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError('Expected a 2D image, got array with {} dimensions instead'.format(image.ndim))

        # Add a third dimension to the image
        image_volumized = np.expand_dims(image, axis=2)

        if pixel_spacing is None:
            voxel_spacing = None
        else:
            voxel_spacing = as_tuple(pixel_spacing, 2, float) + (0.0,)

        # Create 3D patch extractor instance
        self.extractor = PatchExtractor3D(image_volumized, voxel_spacing, pad_value, spline_order, dtype)

    def extract_rect(self, center_pixel, shape, extent=None, rotation_angle=None):
        """Extracts a 2D rectangular patch"""
        center_voxel = (center_pixel[0], center_pixel[1], 0)
        return self.extractor.extract_rect(center_voxel, shape, extent, axis=2, rotation_angle=rotation_angle)
