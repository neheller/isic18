from pathlib import Path
import os

import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input

from task3 import isic_data
from task3 import keras_model_utilities as kmu

_OUTPUT_PATH = Path("/home/helle246/data/isic_models/densenet201")

BATCH_SIZE = 20

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Get callbacks for training/validation
    callbacks = kmu.get_callbacks(_OUTPUT_PATH)

    # Make data pipeline
    trds = isic_data.get_iterator("trn", BATCH_SIZE, 224, preprocess_input)
    vads = isic_data.get_iterator("val", BATCH_SIZE, 224, preprocess_input)

    # Run training
    kmu.run_training_stage(_OUTPUT_PATH, 200,
        np.arange(start=0, stop=100000).tolist(), 0.0001, callbacks, trds, vads,
        DenseNet201
    )
