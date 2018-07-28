from pathlib import Path
import os
import sys

import numpy as np

from task3 import isic_data
from task3 import keras_model_utilities as kmu
from tensorflow.keras.applications.xception import preprocess_input as xception_pp
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pp
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pp
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pp
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_pp

_OUTPUT_PARENT = Path("/home/helle246/data/isic_models")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ensemble_models = [
    [_OUTPUT_PARENT / "inception", inception_pp, 299],
    [_OUTPUT_PARENT / "xception", xception_pp, 299],
    [_OUTPUT_PARENT / "inceptionresnet", inceptionresnet_pp, 299],
    [_OUTPUT_PARENT / "resnet", resnet_pp, 224],
    [_OUTPUT_PARENT / "densenet201", densenet_pp, 224]
]


def validation():
    X, Y = isic_data.get_val_data()

    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)

    # Run for 299 inputs
    inception_predictions = kmu.predict(
        _OUTPUT_PARENT / "inception", inception_pp(X_299)
    )
    del X_299
    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)
    xception_predictions = kmu.predict(
        _OUTPUT_PARENT / "xception", xception_pp(X_299)
    )
    del X_299
    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)
    inceptionresnet_predictions = kmu.predict(
        _OUTPUT_PARENT / "inceptionresnet", inceptionresnet_pp(X_299)
    )
    del X_299
    X_224 = np.zeros((X.shape[0], 224, 224, 3))
    for i in range(0, X.shape[0]):
        X_224[i] = isic_data._preprocess_image_cpu(X[i], False, 224)
    resnet_predictions = kmu.predict(
        _OUTPUT_PARENT / "resnet", resnet_pp(X_224)
    )
    del X_224
    X_224 = np.zeros((X.shape[0], 224, 224, 3))
    for i in range(0, X.shape[0]):
        X_224[i] = isic_data._preprocess_image_cpu(X[i], False, 224)
    densenet_predictions = kmu.predict(
        _OUTPUT_PARENT / "densenet201", densenet_pp(X_224)
    )
    ensemble_predictions = (
        inception_predictions + xception_predictions + inception_predictions +
        resnet_predictions + densenet_predictions
    )/5


    print("INCEPTION:")
    kmu.get_balanced_accuracy(
        np.argmax(inception_predictions, axis=-1), Y
    )
    print("XCEPTION:")
    kmu.get_balanced_accuracy(
        np.argmax(xception_predictions, axis=-1), Y
    )
    print("INCEPTIONRESNET:")
    kmu.get_balanced_accuracy(
        np.argmax(inceptionresnet_predictions, axis=-1), Y
    )
    print("RESNET:")
    kmu.get_balanced_accuracy(
        np.argmax(resnet_predictions, axis=-1), Y
    )
    print("DENSENET:")
    kmu.get_balanced_accuracy(
        np.argmax(densenet_predictions, axis=-1), Y
    )
    print("ENSEMBLE:")
    kmu.get_balanced_accuracy(
        np.argmax(ensemble_predictions, axis=-1), Y
    )
    np.save("ensemble_validation_predictions.npy", ensemble_predictions)
    np.save("validation_y.npy", Y)



def testing():
    X, fnames = isic_data.get_test_data()

    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)

    # Run for 299 inputs
    inception_predictions = kmu.predict(
        _OUTPUT_PARENT / "inception", inception_pp(X_299)
    )
    del X_299
    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)
    xception_predictions = kmu.predict(
        _OUTPUT_PARENT / "xception", xception_pp(X_299)
    )
    del X_299
    X_299 = np.zeros((X.shape[0], 299, 299, 3))
    for i in range(0, X.shape[0]):
        X_299[i] = isic_data._preprocess_image_cpu(X[i], False, 299)
    inceptionresnet_predictions = kmu.predict(
        _OUTPUT_PARENT / "inceptionresnet", inceptionresnet_pp(X_299)
    )
    del X_299
    X_224 = np.zeros((X.shape[0], 224, 224, 3))
    for i in range(0, X.shape[0]):
        X_224[i] = isic_data._preprocess_image_cpu(X[i], False, 224)
    resnet_predictions = kmu.predict(
        _OUTPUT_PARENT / "resnet", resnet_pp(X_224)
    )
    del X_224
    X_224 = np.zeros((X.shape[0], 224, 224, 3))
    for i in range(0, X.shape[0]):
        X_224[i] = isic_data._preprocess_image_cpu(X[i], False, 224)
    densenet_predictions = kmu.predict(
        _OUTPUT_PARENT / "densenet201", densenet_pp(X_224)
    )

    np.save("inception_predictions.npy", inception_predictions)
    np.save("xception_predictions.npy", xception_predictions)
    np.save("inceptionresnet_predictions.npy", inceptionresnet_predictions)
    np.save("resnet_predictions.npy", resnet_predictions)
    np.save("densenet_predictions.npy", densenet_predictions)
    np.save("filenames.npy", fnames)


    # print("INCEPTION:")
    # kmu.get_balanced_accuracy(
    #     np.argmax(inception_predictions, axis=-1), Y
    # )
    # print("XCEPTION:")
    # kmu.get_balanced_accuracy(
    #     np.argmax(xception_predictions, axis=-1), Y
    # )
    # print("INCEPTIONRESNET:")
    # kmu.get_balanced_accuracy(
    #     np.argmax(inceptionresnet_predictions, axis=-1), Y
    # )
    # print("RESNET:")
    # kmu.get_balanced_accuracy(
    #     np.argmax(resnet_predictions, axis=-1), Y
    # )
    # print("DENSENET:")
    # kmu.get_balanced_accuracy(
    #     np.argmax(densenet_predictions, axis=-1), Y
    # )




if __name__ == '__main__':
    validation()
    # testing()
