# ==============================================================================
"""A script to iterate over a directory and resize all of the jpg images to
_NEW_SIZE x _NEW_SIZE

Will iterate into a directory's children
"""

from pathlib import Path
import sys

from progress.bar import Bar
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

_NEW_SIZE = 350


if __name__ == '__main__':
    image_directory = Path(sys.argv[1]).resolve()

    num_ims = sum([1 for _ in image_directory.glob("**/ISIC_*.jpg")])

    X = np.zeros((num_ims, _NEW_SIZE, _NEW_SIZE, 3), np.uint8)
    filenames = []

    bar = Bar("Converting...", max=num_ims)
    for i, image in enumerate(image_directory.glob("**/ISIC_*.jpg")):
        bar.next()
        im_arr = imread(str(image))
        resized_array = resize(im_arr, (_NEW_SIZE, _NEW_SIZE))
        im_to_write = (resized_array*255).astype(np.uint8)
        X[i,:,:,:] = im_to_write
        filenames = filenames + [image.stem]
        # np.save(image.parent / (image.stem + ".npy"), im_to_write)
    bar.finish()
    np.save("/home/helle246/Desktop/X.npy", X)
    np.save("/home/helle246/Desktop/fnames.npy", np.array(filenames))
