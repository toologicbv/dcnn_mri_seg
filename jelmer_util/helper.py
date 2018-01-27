import os
import numpy as np
import SimpleITK as sitk


def load_mhd_to_npy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    return np.swapaxes(sitk.GetArrayFromImage(image), 0, 2), spacing


def loadImageDir(imagefiles, nclass=2):
    print(imagefiles)

    # Images is a list of 256x256 2D images
    images = []
    # Labels is a list of 256x256 2D images
    labels = []
    processed = 0

    classcount = np.zeros((nclass))

    # Iterate over training images
    for f in imagefiles:
        print('Loading ' + str(processed) + '/' + str(len(imagefiles)))
        print(f)
        processed = processed + 1
        reffile = f.replace('image', 'label')
        if 'brain' in f or 'breast' in f:
            reffile = reffile.replace('nii', 'mhd')
        # If reference file exists
        if os.path.isfile(reffile):
            # Load image file
            image, spacing = load_mhd_to_npy(f)
            # Load reference file
            ref, spacing = load_mhd_to_npy(reffile)

            imame = np.mean(image)
            imast = np.std(image)
            image = (image-imame)/imast
            print('Mean intensity ' + str(np.mean(image)))
            if 'cor' in f:
                image += 1024

            padwidth = 65 # 66 # 33 # 65 #33 # 65 # 129 #65 # 129 # 17 # 33 # 65 # 129 # 8 # 65

            for cl in range(nclass):
                classcount[cl] += np.sum(ref==cl)

            intorfloat = 'float32'
            if 'pancreas' in reffile:
                intorfloat = 'int16'
            for z in range(image.shape[2]):
                # print('Add axial')
                laslice = np.squeeze(ref[:,:,z])
                imslice = np.squeeze(image[:,:,z])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)

            # if not 'pancreas' in reffile:
            for y in range(image.shape[1]):
                # print('Add axial')
                laslice = np.squeeze(ref[:,y,:])
                imslice = np.squeeze(image[:,y,:])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)

            for x in range(image.shape[0]):
                # print('Add axial')
                laslice = np.squeeze(ref[x,:,:])
                imslice = np.squeeze(image[x,:,:])

                # PAD IMAGE
                for rots in range(4):
                    section = np.pad(imslice, padwidth, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    images.append(section)
                    section = laslice
                    # np.pad(laslice, padwidth/2, 'constant', constant_values=(0,)).astype(intorfloat) # 65 # 129
                    labels.append(section)
                    imslice = np.rot90(imslice)
                    laslice = np.rot90(laslice)

    return images, labels, classcount


def generateBatch2D(images, labels, nsamp=10, nclass=3, classcount=(1,1)):
    ps = 201 # 137 # 201 # 329 # 89 # 105 # 137 # 201 # 329 # 259 + 70 = 329 # 17 + 70 = 87 # 131 + 70 = 201 # 329 # 201
    ss = 70
    # mw = (259-1)/2

    # batch image patch
    batch_im = np.zeros((nsamp, 1, ps, ps))
    # batch reference image for the different classes (so for each reference class 1 image)
    batch_la = np.zeros((nsamp, nclass, ss+1, ss+1))
    # the "complete" reference image - containing all class labels
    class_im = np.zeros((1, nsamp, ss+1, ss+1))

    for ns in range(nsamp):
        ind = np.random.randint(0, len(images))
        imageim = images[ind]

        labelim = labels[ind]

        offx = np.random.randint(0, imageim.shape[0]-ps)
        offy = np.random.randint(0, imageim.shape[1]-ps)

        imageim = imageim[offx:offx+ps, offy:offy+ps]

        batch_im[ns, 0, :, :] = imageim

        labelim = labelim[offx:offx+ss+1, offy:offy+ss+1]

        class_im[0, ns, :, :] = labelim

        for cl in range(nclass):
            batch_la[ns, cl, :, :] = (labelim==cl).astype('int16')

    batch_mask = np.zeros((1, nsamp, ss+1, ss+1), dtype='float32')

    return batch_im, class_im, batch_la, batch_mask