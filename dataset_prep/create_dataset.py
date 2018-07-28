# ==============================================================================
"""A script to accept a structured directory of images and create a
tfrecord

The goal of this is to get experience following Google's best practices in this
area
    - jk Google hasn't given any best practices nor documentation of tf records.

The input directory will be structured by class, including a class for unlabeled
data. The images will be in JPG format (ISIC archive) and of all different
sizes.

The images will be reshaped to 512x512 and stored as examples in individual
tfrecord files sorted by class with the following features:
    image: 512x512x3 array of uint8s
    label: which class they belong to (-1 through 6, -1 meaning unlabeled)
    name: the name of the jpg that the image came from (stem)

The reason I'm not globbing many images together in a record is for ease and
effectiveness of shuffling their order at training time.
    dataset.shuffle() is extremely slow, memory inefficient, and ineffective

NOTE: Python >= 3.0
"""

from pathlib import Path
import argparse
import six

import tensorflow as tf
from skimage import io
from skimage.transform import resize
import numpy as np
from progress.bar import Bar

_NEW_HEIGHT = 512
_NEW_WIDTH = 512


desc_string = "Convert directory of images into TensorFlow's TFRecord format"
parser = argparse.ArgumentParser(description=desc_string)
parser.add_argument(
    '-id', '--input_directory', type=str, required=True,
    help="Directory with images in subdirs named with their respective label"
) # Takes None if not specified
parser.add_argument(
    '-od', '--output_directory', type=str, required=True,
    help="Directory in which to write TFRecords"
)
args = parser.parse_args()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _get_image(fpath):
    return (resize(
        io.imread(str(fpath)), (_NEW_HEIGHT, _NEW_WIDTH)
    )*255).astype(np.uint8)


def write_class_records(directory, label, record_path):
    if not record_path.exists()
        record_path.mkdir()
    filenames = [f for f in directory.glob("*.jpg")]
    bar = Bar("Working on %s ..." % str(record_path), max=len(filenames))
    for f in filenames:
        writer = tf.python_io.TFRecordWriter(str(record_path / (f.stem+".tfrecord")))
        bar.next()
        expl = tf.train.Example(
            features=tf.train.Features(feature={
                "filename": _bytes_feature(str.encode(f.name)),
                "image": _bytes_feature(_get_image(f).tostring()),
                "label": _int64_feature(label)
            })
        )
        writer.write(expl.SerializeToString())
        writer.close()
    bar.finish()


def get_label(directory_name):
    try:
        return int(directory_name)
    except ValueError:
        return -1


if __name__ == '__main__':
    input_directory = Path(args.input_directory).resolve()
    od = Path(args.output_directory)
    output_directory = od.parent.resolve() / od.name
    if not output_directory.exists():
        output_directory.mkdir()

    for d in input_directory.glob("*"):
        record_path = output_directory / (d.name)
        label = get_label(d.name)
        write_class_records(d, label, record_path)
