# ==============================================================================
"""A script to split an organized dataset (under 0, 1, 2, ...) into
training and validation sets
"""

from pathlib import Path
import argparse
import shutil
from random import shuffle

desc_string = "Split your organized dataset into training and validation sets"
parser = argparse.ArgumentParser(description=desc_string)
parser.add_argument(
    '-id', '--input_directory', type=str, required=True,
    help="Directory with images in subdirs named with their respective label"
)
parser.add_argument(
    '-od', '--output_directory', type=str, required=True,
    help="Directory in which to write TFRecords"
)
parser.add_argument(
    '-nc', '--num_classes', type=int, default=2,
    help="Number of classes to split, 0, 1, 2, ..., nc - 1"
)
parser.add_argument(
    '-tp', '--training_proportion', type=float, default=0.85,
    help="Directory with images in subdirs named with their respective label"
)
args = parser.parse_args()


def split_class(idr, odr, i, tp):
    input_class_dir = idr / str(i)
    assert input_class_dir.exists() and input_class_dir.is_dir()
    output_trn_dir = odr / "trn" / str(i)
    output_val_dir = odr / "val" / str(i)
    if not output_trn_dir.exists():
        output_trn_dir.mkdir()
    if not output_val_dir.exists():
        output_val_dir.mkdir()

    filenames = [f for f in input_class_dir.glob("*")]
    shuffle(filenames)
    num = len(filenames)

    for f in filenames[:int(tp*num)]:
        shutil.move(str(f), str(output_trn_dir / f.name))
    for f in filenames[int(tp*num):]:
        shutil.move(str(f), str(output_val_dir / f.name))


def train_test_split(idr, odr, nc, tp):
    if not odr.exists():
        odr.mkdir()
    if not (odr / "trn").exists():
        (odr / "trn").mkdir()
    if not (odr / "val").exists():
        (odr / "val").mkdir()

    for i in range(0, nc):
        split_class(idr, odr, i, tp)


if __name__ == '__main__':
    input_directory = Path(args.input_directory).resolve()
    output_directory = Path(args.output_directory)
    output_directory = output_directory.parent.resolve() / output_directory.name

    num_classes = args.num_classes
    training_proportion = args.training_proportion

    summary_string = """
Running traing_test_split with:
    input_directory = {0},
    output_directory = {1},
    num_classes = {2},
    training_proportion = {3}

Data will be moved from the input directory to the output directory and
join or overwrite any data that might exist. Would you like to continue?

Press [ENTER] to continue, or [ctrl+C] to quit
    """
    print(
        summary_string.format(
            str(input_directory), str(output_directory), str(num_classes),
            str(training_proportion)
        )
    )
    input()
    train_test_split(
        input_directory, output_directory, num_classes, training_proportion
    )
