# ==============================================================================
"""A script to sort the ISIC images by their class

NOTE: python >= 3.0
"""

import shutil
from pathlib import Path
import csv

import numpy as np
from progress.bar import Bar


def get_gt_csv_reader(idir):
    """Get a csv reader for the ground truth csv file given the directory
    containing the two unzipped folders from the challenge website"""
    gt_dir = [i for i in idir.glob("*GroundTruth*") if i.is_dir()][0]
    gt_fl = [i for i in gt_dir.glob("*.csv")][0]
    fh = gt_fl.open('r')
    reader = csv.reader(fh)
    next(reader) # skip header
    num_images = sum([1 for row in reader])
    fh.seek(0)
    next(reader)
    return reader, num_images


if __name__ == '__main__':
    idir = Path("/home/helle246/data/ISIC18/task3")
    odir = Path("/home/helle246/data/ISIC18/task3_sorted2")
    im_dir = [i for i in idir.glob("*Input*")][0]

    if not odir.exists():
        odir.mkdir()
        for i in range(0,7):
            (odir / str(i)).mkdir()
    else:
        raise ValueError("Output directory exists {0}".format(str(odir)))


    reader, num_images = get_gt_csv_reader(idir)
    bar = Bar('Copying...', max=num_images)
    for row in reader:
        bar.next()
        corresponding_image = [i for i in im_dir.glob("*"+row[0]+"*")][0]
        label = np.argmax([i for i in map(lambda x: int(float(x)), row[1:])])
        target = odir / str(label)
        shutil.copy(
            str(corresponding_image), str(target / corresponding_image.name)
        )
    bar.finish()
