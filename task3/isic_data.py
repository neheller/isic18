# ==============================================================================
"""A module to be imported by a main module in order to feed balanced isic data
to an estimator training on the objective for task 3 of the 2018 ISIC challenge

Dataset takes the form of individual tfrecord files for each image. Filenames
are scraped from hard-coded data directory using pathlib and shuffled and
repeated using native python functions. Then a vanilla tf.data.TFRecordDataset
is instantiated.
"""

from pathlib import Path
from random import shuffle
import math

import tensorflow as tf
import keras
import numpy as np
from skimage.transform import resize, rotate
from PIL import ImageEnhance, Image

# from task3.resnet import resnet_run_loop

_DATA_DIR = Path("/home/helle246/data/isic_2018_records_separate")
_GEN_DATA_DIR = Path("/home/helle246/data/ISIC18/task3_split")
_TST_DATA_LOCATION = Path("/home/helle246/data/isic_2018_test/X.npy")
_TST_FNAMES_LOCATION = Path("/home/helle246/data/isic_2018_test/fnames.npy")
_VAL_DATA_LOCATION = Path("/home/helle246/data/isic_2018_val/X.npy")
_VAL_LABELS_LOCATION = Path("/home/helle246/data/isic_2018_val/Y.npy")
_NUM_CLASSES = 7
_SHUFFLE_BUFFER = 10000
_EXAMPLES_PER_EPOCH = 10000
_BRIGHTNESS_MAX_DELTA = 0.3
_CONTRAST_MAX_DELTA = 0.7


def _preprocess_image_cpu(image, is_training, new_size):
    dim = image.shape[1]
    image = image.astype(np.float32)/256.0
    if not is_training:
        # Crop center and return
        image = resize(image, (new_size+30, new_size+30))
        return image[15:-15, 15:-15, :]*256.0
    # else

    # Get random parameters of this augmentation
    angle = np.random.uniform(low=0.0, high=360.0)
    intermediate_dim = int(np.random.normal(
        loc=new_size+30, scale=15.0
    ))
    if intermediate_dim < new_size:
        intermediate_dim = new_size
    rng = intermediate_dim - new_size
    starts = [0,0]
    if rng != 0:
        starts = np.random.randint(low=0, high=rng, size=(2,))
    flip_lr = np.random.uniform() > 0.5
    flip_ud = np.random.uniform() > 0.5
    rnd_contrast = np.random.uniform(
        low=1.0-_CONTRAST_MAX_DELTA, high=1.0+_CONTRAST_MAX_DELTA
    )
    rnd_brightness = np.random.uniform(
        low=1.0-_BRIGHTNESS_MAX_DELTA, high=1.0+_BRIGHTNESS_MAX_DELTA
    )

    # Perform augmentation operations
    image = resize(image, (intermediate_dim, intermediate_dim))
    image = rotate(image, angle=angle)
    image = image[starts[0]:starts[0]+new_size, starts[1]:starts[1]+new_size, :]
    if flip_ud:
        image = np.flip(image, axis=0)
    if flip_lr:
        image = np.flip(image, axis=1)
    image = Image.fromarray((image*256).astype(np.uint8))
    contrast_changer = ImageEnhance.Contrast(image)
    image = contrast_changer.enhance(rnd_contrast)
    brightness_changer = ImageEnhance.Brightness(image)
    image = brightness_changer.enhance(rnd_brightness)
    image = np.array(
        image.getdata()
    ).astype(np.float32).reshape((new_size, new_size, 3))/256.0
    return image*256.0


def _preprocess_image(image, is_training, new_size, ppfn):
    image = tf.cast(image, tf.float32)/256.0
    if not is_training:
        image = tf.image.resize_images(image, (new_size+30, new_size+30))
        return ppfn(tf.image.resize_image_with_crop_or_pad(
            image, new_size, new_size
        )*256.0)
    # else

    # Get random parameters of this augmentation
    angle = tf.random_uniform((), minval=0, maxval=2*math.pi, dtype=tf.float32)
    dim = tf.cast(
        tf.random_normal((), mean=new_size+30.0, stddev=15.0), tf.int32
    )
    dim = tf.cond(tf.greater(dim, new_size), lambda: dim, lambda: new_size)

    # Perform augmentation operations
    image = tf.image.resize_images(image, (dim, dim))
    image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
    image = tf.random_crop(
        image,
        [new_size, new_size, 3]
    )
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(
            image, max_delta=_BRIGHTNESS_MAX_DELTA
    )
    image = tf.image.random_contrast(
        image,
        lower=1.0-_CONTRAST_MAX_DELTA, upper=1.0+_CONTRAST_MAX_DELTA
    )
    return ppfn(image*256.0)


def parse_record(raw_record, is_training, new_size=224, ppfn=None):
    """Parse a single instance record

    Returns:
        Tuple with processed image tensor and sparse label tensor (label index)
    """
    feature_map = {
        'filename': tf.FixedLenFeature([], dtype=tf.string),
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    features = tf.parse_single_example(raw_record, feature_map)
    image_bytes = tf.decode_raw(features["image"], tf.uint8)
    # filename = tf.cast(features["filename"], tf.string)
    image = tf.reshape(
        image_bytes, (512, 512, 3)
    )
    label = tf.cast(features["label"], tf.int32)

    image = _preprocess_image(
        image=image,
        is_training=is_training,
        new_size=new_size,
        ppfn=ppfn
    )

    return image, tf.one_hot(label, 7)


def _fname_repeat(lst, target_length):
    current_length = len(lst)
    return [lst[i%current_length] for i in range(0, target_length)]


def get_filenames(typ, data_dir, extension='tfrecord'):
    print("GET FILENAMES CALLED")
    fnames_by_class = [
        [str(p) for p in (data_dir / typ / str(c)).glob("ISIC_*." + extension)]
        for c in range(0, _NUM_CLASSES)
    ]
    fname_lengths = [len(l) for l in fnames_by_class]
    max_len = max(fname_lengths)
    fnames = []
    for i in range(0, _NUM_CLASSES):
        fnames = fnames + _fname_repeat(fnames_by_class[i], max_len)
    shuffle(fnames)

    return fnames


def get_iterator(typ, batch_size, new_size=299, ppfn=None):
    fnames = get_filenames(typ, _DATA_DIR)
    shuffle(fnames)
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    dataset = dataset.shuffle(buffer_size=1000000)
    # dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda x: parse_record(x, typ=="trn", new_size=new_size, ppfn=ppfn),
            batch_size, drop_remainder=True
        )
    )
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


# Holy slow...
class ISICGenerator(keras.utils.Sequence):

    def __init__(self, typ, dim, batch_size):
        self.typ = typ
        self.fnames = get_filenames(typ, _GEN_DATA_DIR, extension='npy')

        self.dim = dim
        self.batch_size = batch_size

        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.fnames) / self.batch_size))


    def __getitem__(self, index):
        if index % 100 == 0:
            shuffle(self.fnames)
        names = self.fnames[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.get_images(names)
        return X, Y


    def on_epoch_end(self):
        shuffle(self.fnames)


    def get_images(self, names):
        X = np.zeros((self.batch_size, self.dim, self.dim, 3), np.float32)
        Y = np.zeros((self.batch_size,), np.int64)

        # Generate data
        for i, name in enumerate(names):
            # Store sample
            X[i,:,:,:] = _preprocess_image_cpu(np.load(name), self.typ=="trn", 299)

            # Store class
            Y[i] = int(Path(name).parent.name)

        return X, Y


def get_test_data():
    return np.load(str(_TST_DATA_LOCATION)), np.load(str(_TST_FNAMES_LOCATION))

def get_val_data():
    return np.load(str(_VAL_DATA_LOCATION)), np.load(str(_VAL_LABELS_LOCATION))
