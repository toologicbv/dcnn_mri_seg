import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import PIL as pil
import time
# import scipy.interpolate as scip
# import scipy.stats as scst
import sys
import scipy.ndimage.interpolation as spin
import scipy

def rotate_scipy_affine(image, alpha, beta, gamma, scale = 1.0):
    margin = np.int(np.max(np.array(image.shape))/2)
    #image = np.pad(image, margin, mode = 'constant', constant_values = -1024)

    alpha = (np.squeeze(alpha) / 180.) * np.pi
    beta = (np.squeeze(beta) / 180.) * np.pi
    gamma = (np.squeeze(gamma) / 180.) * np.pi

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    rotMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb - sa * sg],
                          [cb * sg, ca * cg + sa * sb * sg, -1. * cg * sa + ca * sb * sg],
                          [-1 * sb, cb * sa, ca * cb]])

    # Center in input image
    c_in = 0.5*np.array(image.shape)
    # Transformed center
    c_out = scale*0.5*np.array(image.shape)
    offset = c_in - c_out.dot(rotMatrix)
    print(offset)
    #e.shape[0]/2), -1*(image.shape[1]/2), -1*(image.shape[2]/2)]
    return spin.affine_transform(image, rotMatrix.T, offset = offset, output_shape=(scale*image.shape[0], scale*image.shape[1], scale*image.shape[2]))

def rotate_lena():
    src=scipy.misc.lena()
    c_in=0.5*np.array(src.shape)
    c_out=np.array((256.0,256.0))
    for i in xrange(0,7):
        a=i*15.0*np.pi/180.0
        transform=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        offset=c_in-c_out.dot(transform)
        dst=scipy.ndimage.interpolation.affine_transform(
            src,transform,order=2,offset=offset,output_shape=(512,512),cval=0.0,output=np.float32
        )
        plt.subplot(1,7,i+1);plt.axis('off');plt.imshow(dst,cmap=plt.cm.gray)
    plt.show()

def rotate_scipy_affine_test(filename):
    print('Testing rotation')
    nim = load_mhd_to_npy(filename)
    nim = np.swapaxes(nim, 0, 2)
    nim = rotate_scipy_affine(nim, 0.0, 0.0, 90.0, scale = 2.0)
    #nim = np.swapaxes(nim, 0, 2)
    sim = sitk.GetImageFromArray(nim)
    write_mhd(sim, r'U:\users\Jelmer\MEDIA2016\tmp\out.mhd')
    #rotate_lena()

    nim = rotate_scipy_affine(nim, 0.0, 0.0, -90.0, scale = 0.5)
    #nim = np.swapaxes(nim, 0, 2)
    sim = sitk.GetImageFromArray(nim)
    write_mhd(sim, r'U:\users\Jelmer\MEDIA2016\tmp\outinv.mhd')

def rotate_scipy(image, alpha, beta, gamma):
    axes = (1, 0)
    angle = 0.0
    # Rotate around x-axis
    if np.abs(alpha) > 0.0:
        axes = (1, 2)
        angle = alpha
    else:
        if np.abs(beta) > 0.0:
            axes = (0, 2)
            angle = beta
        else:
            if np.abs(gamma) > 0.0:
                axes = (0, 1)
                angle = gamma
    return spin.rotate(image, angle, axes)


def rotate_scipy_test(filename, alpha, beta, gamma):
    nim = load_ii_to_npy(filename)
    slice = np.squeeze(nim[:, :, 100])
    fig = plt.figure()
    suba = fig.add_subplot(1, 2, 1)
    plt.imshow(slice)
    # Angle in degrees
    rot = rotate_scipy(nim, alpha, beta, gamma)
    subb = fig.add_subplot(1, 2, 2)
    plt.imshow(np.squeeze(rot[:, :, 100]))
    plt.show()
    plt.draw()


def rotate(image, alpha, beta, gamma, spacingx, spacingy, spacingz):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    rotMatrix = np.array([cb * cg, cg * sa * sb - ca * sg, ca * cg * sb - sa * sg,
                          cb * sg, ca * cg + sa * sb * sg, -1. * cg * sa + ca * sb * sg,
                          -1 * sb, cb * sa, ca * cb])

    targetspacing = np.array([spacingx, spacingy, spacingz])
    image = sitk.Cast(image, sitk.sitkFloat32)
    sourcespacing = np.array([image.GetSpacing()])
    sourcesize = np.array([image.GetSize()])
    scale = sourcespacing / targetspacing
    image.SetOrigin((0.0, 0.0, 0.0))
    targetsize = sourcesize * scale
    targetsize = targetsize.astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(targetspacing)
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(tuple(targetsize[0]))

    # User versor transform
    rotation = sitk.VersorTransform()
    rotation.SetMatrix(rotMatrix)
    rotation_center = (sourcespacing[0][0] * sourcesize[0][0] / 2, sourcespacing[0][1] * sourcesize[0][1] / 2,
                       sourcespacing[0][2] * sourcesize[0][2] / 2)
    print(rotation_center)
    rotation.SetCenter(rotation_center)
    resampler.SetTransform(rotation)
    out = resampler.Execute(image)
    out = sitk.Cast(out, sitk.sitkInt16)
    return out


def resample(image, spacingx, spacingy, spacingz, interpolator = sitk.sitkLinear):
    targetspacing = np.array([spacingx, spacingy, spacingz])
    image = sitk.Cast(image, sitk.sitkFloat32)
    sourcespacing = np.array([image.GetSpacing()])
    sourcesize = np.array([image.GetSize()])
    scale = sourcespacing / targetspacing
    image.SetOrigin((0.0, 0.0, 0.0))
    targetsize = sourcesize * scale
    targetsize = targetsize.astype(int)
    transform = sitk.Transform(3, sitk.sitkScale)
    transform.SetParameters((1.0, 1.0, 1.0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(targetspacing)
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(tuple(targetsize[0]))
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    out = resampler.Execute(image)
    out = sitk.Cast(out, sitk.sitkInt16)
    return out


def write_mhd(image, filename):
    sitk.WriteImage(image, filename)


def load_ii(filename):
    path = ''
    with open(filename, 'r') as f:
        for line in f:
            kv = line.split('=')
            k = kv[0]
            v = kv[1].rstrip()
            if k == 'path':
                path = v
    print(path)
    if path != '':
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    else:
        return 0


def rotate_voxel(nim, rx, ry, x, y, z, linear, rotMatrix):
    rotpoint = np.dot(rotMatrix, np.array([rx - x, ry - y, 0, 1.0]))  # rz and z are the same, hence third argument = 0
    nx = rotpoint[0] + x  # + np.float(margin*2+x)
    ny = rotpoint[1] + y  # + np.float(margin*2+y)
    nz = rotpoint[2] + z  # + np.float(margin*2+z)

    # Trilinear interpolation
    if linear:
        bx = np.int(nx)
        by = np.int(ny)
        bz = np.int(nz)
        tx = bx + 1
        ty = by + 1
        tz = bz + 1
        xd = nx - bx
        yd = ny - by
        zd = nz - bz
        C000 = nim[bx, by, bz]
        C001 = nim[bx, by, tz]
        C010 = nim[bx, ty, bz]
        C011 = nim[bx, ty, tz]
        C100 = nim[tx, by, bz]
        C101 = nim[tx, by, tz]
        C110 = nim[tx, ty, bz]
        C111 = nim[tx, ty, tz]
        C00 = (1 - xd) * C000 + xd * C100
        C01 = (1 - xd) * C001 + xd * C101
        C10 = (1 - xd) * C010 + xd * C110
        C11 = (1 - xd) * C011 + xd * C111
        C0 = (1 - yd) * C00 + yd * C10
        C1 = (1 - yd) * C01 + yd * C11
        return (1 - zd) * C0 + zd * C1
    else:
        nx = np.int(0.5 + nx)
        ny = np.int(0.5 + ny)
        nz = np.int(0.5 + nz)
        return nim[nx, ny, nz]


def draw_sample_3D(nim, pw, x, y, z):
    margin = (pw - 1) / 2
    return nim[x - margin:x + margin + 1, y - margin:y + margin + 1, z - margin:z + margin + 1]


def draw_sample(nim, pw, x, y, z, rotation=False, linear=True, triplanar=False, direction='Axial'):
    if not rotation:
        if triplanar:
            return draw_sample_triplanar(nim, pw, x, y, z)
        else:
            return draw_sample_no_rot(nim, pw, x, y, z, direction)
    else:
        # Constrain angles between 0 and 90 degrees
        alpha = (np.squeeze(np.float(np.random.randint(0, 90, 1))) / 180.) * np.pi
        beta = (np.squeeze(np.float(np.random.randint(0, 90, 1))) / 180.) * np.pi
        gamma = (np.squeeze(np.float(np.random.randint(0, 90, 1))) / 180.) * np.pi

        if triplanar:
            keepaxis = np.random.randint(1, 3)
            print(keepaxis)
            if keepaxis == 1:
                beta = 0.0
                gamma = 0.0
            if keepaxis == 2:
                alpha = 0.0
                gamma = 0.0
            if keepaxis == 3:
                alpha = 0.0
                beta = 0.0

        # Fix angles to 0 or 90 degrees
        # alpha = np.round(np.random.rand()) * 0.5 * np.pi
        # beta = np.round(np.random.rand()) * 0.5 * np.pi
        # gamma = np.round(np.random.rand()) * 0.5 * np.pi
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        rotMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb - sa * sg, 0.],
                              [cb * sg, ca * cg + sa * sb * sg, -1. * cg * sa + ca * sb * sg, 0.],
                              [-1 * sb, cb * sa, ca * cb, 0],
                              [0, 0, 0, 1.]])

        start = time.time()
        margin = (pw - 1) / 2
        rotpatch = np.zeros((pw, pw))
        for rx in range(x - margin, x + margin + 1):
            for ry in range(y - margin, y + margin + 1):
                rotpatch[rx - (x - margin), ry - (y - margin)] = rotate_voxel(nim, rx, ry, x, y, z, linear, rotMatrix)
        return rotpatch
        # if write_im:
        #    plt.imshow(rotpatch, vmin = -285, vmax = 465, cmap = plt.cm.gray)
        #    if linear:
        #        plt.savefig(r'U:\users\Jelmer\MEDIA2016\rotate\\' + str(it) + '_linear.jpg')
        #    else:
        #        plt.savefig(r'U:\users\Jelmer\MEDIA2016\rotate\\' + str(it) + '_nearest.jpg')
        # print('Took ' + str(time.time() - start) + ' s')


def draw_sample_triplanar(nim, pw, x, y, z):
    orientation = np.random.randint(low=0, high=3)
    margin = (pw - 1) / 2
    #print(str(orientation))
    if orientation == 0:
        return draw_sample_no_rot(nim, pw, x, y, z)
    elif orientation == 1:
        return np.squeeze(nim[x, y - margin:y + margin + 1, z - margin:z + margin + 1])
    elif orientation == 2:
        return np.squeeze(nim[x - margin:x + margin + 1, y, z - margin:z + margin + 1])


def draw_sample_no_rot(nim, pw, x, y, z, direction = 'Axial'):
    #print(direction)
    margin = (pw - 1) / 2
    if direction == 'Axial':
        return nim[x - margin:x + margin + 1, y - margin:y + margin + 1, z]
    elif direction == 'Sagittal':
        return nim[x, y - margin:y + margin + 1, z - margin:z + margin + 1]
    elif direction == 'Coronal':
        return nim[x - margin:x + margin + 1, y, z - margin:z + margin + 1]
    else:
        print('Invalid direction!')
        return 0

# Input 3-column coordinates
def CartesianToSpherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def try_rotate(filename):
    np.random.seed(seed=1)
    linear = True
    saveit = False
    image = sitk.ReadImage(filename)
    nim = sitk.GetArrayFromImage(image)
    nim = np.swapaxes(nim, 0, 2)
    # imslice = np.transpose(np.squeeze(nim[:,:,100]))
    pw = 45
    margin = (pw - 1) / 2
    x = 295
    y = 274
    z = 267
    # patch = imslice[x-margin:x+margin+1,y-margin:y+margin+1]
    # plt.imshow(patch)
    # print(patch.shape)
    nim = np.pad(nim, margin * 2, 'reflect')

    start = time.time()
    npatch = 20000
    for rots in range(npatch):
        draw_sample(nim, margin, pw, x, y, z, linear, rots, saveit)
    print('Took ' + str((time.time() - start) / npatch) + ' s per patch, ' + str((time.time() - start)) + ' s in total')


def make_percentile_image(filename):
    image = sitk.ReadImage(filename)
    nim = sitk.GetArrayFromImage(image)
    nim = np.swapaxes(nim, 0, 2)
    nimslice = np.squeeze(nim[:, :, 100])
    nimslice = np.squeeze(np.reshape(nimslice, (1, nimslice.shape[0] * nimslice.shape[1])))
    print(nimslice.shape)
    pimslice = [scst.percentileofscore(nimslice, i) for i in nimslice]
    pimslice = np.reshape(pimslice, (nim.shape[0], nim.shape[1]))
    plt.imshow(pimslice)


def ind2sub(size, index):
  sub = np.zeros((3))
  print(str(sub))
  sub[0] = np.ceil(float(index)/float(size[1]*size[2]))
  index = index-(sub[0]-1)*(size[1]*size[2])
  sub[1] = np.ceil(index/(size[2]))
  sub[2] = index-(sub[1]-1)*(size[2])
  return sub

def main(filename):
    image = load_ii(filename)
    plt.imshow(sitk.GetArrayFromImage(image)[:, :, 100])
    print(image.GetSize())
    # write_mhd(image, r'U:\users\Jelmer\MEDIA2016\mhd\1_or.mhd')
    # image = resample(image, 1.0, 1.0, 1.0)
    image = rotate(image, 15, 15, 15, 0.45, 0.45, 0.45)
    print(image.GetSize())
    write_mhd(image, r'U:\users\Jelmer\MEDIA2016\mhd\1_rot.mhd')


def load_ii_to_npy(filename):
    image = load_ii(filename)
    print(image.GetSize())
    image = resample(image, 0.45, 0.45, 0.45)
    print(image.GetSize())
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

def load_npy_to_npy(filename):
    return np.load(filename) #, mmap_mode='r')

def load_mhd_to_npy(filename):
    image = sitk.ReadImage(filename)
    #print(image.GetSize())
    spacing = image.GetSpacing()
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2), spacing


def to_npy_file(filename, outfile):
    image = load_ii(filename)
    print(filename)
    print(image.GetSize())
    image = resample(image, 0.45, 0.45, 0.45)
    print(image.GetSize())
    # write_mhd(image, r'U:\users\Jelmer\MEDIA2016\mhd\1.mhd')
    np.save(outfile, np.swapaxes(sitk.GetArrayFromImage(image), 0, 2))


if __name__ == "__main__":
    print(sys.argv)
    #rotate_scipy(sys.argv[1], 0.0, 0.0, 90.0)
    rotate_scipy_affine_test(sys.argv[1])
    # main(r'U:\users\Jelmer\MEDIA2016\ii\100.ii')
    # if len(sys.argv) > 1:
    #     print(sys.argv[1])
    #     if len(sys.argv) > 2:
    #         print(sys.argv[2])
    # to_npy_file(sys.argv[1], sys.argv[2])
    # try_rotate(sys.argv[1]) #r'U:\users\Jelmer\DLCTA\ImagesResampled\5.mhd') #
    # try_rotate(r'U:\users\Jelmer\DLCTA\ImagesResampled\5.mhd')
    # make_percentile_image(r'U:\users\Jelmer\DLCTA\ImagesResampled\5.mhd')

    # sys.argv[1:])