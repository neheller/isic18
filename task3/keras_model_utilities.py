from pathlib import Path
import os
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def restore_from_latest(output_path):
    if not output_path.exists():
        output_path.mkdir()

    model = None
    for json_file in output_path.glob("*.json"):
        if model is None:
            json_string = json_file.open('r').read()
            model = keras.models.model_from_json(json_string)
        else:
            raise IOError("More than one json file in directory...")

    if model is not None:
        weights_files = [wf for wf in output_path.glob("*.h5")]
        if len(weights_files) == 0:
            raise IOError("No weights to restore from at {0}".format(str(output_path)))
        weights_accs = [float(wf.name[-8:-3]) for wf in weights_files]
        best_weights_file = weights_files[np.argmax(weights_accs)]
        model.load_weights(str(best_weights_file))
        return model, json_file, best_weights_file
    else:
        raise IOError("No model to restore from at {0}".format(str(output_path)))

def get_model(application_fn, output_path):
    try:
        model, model_file, weights_file = restore_from_latest(output_path)
        epoch = int(weights_file.name[14:17])
        print(
            ("Restored from graph defined by {0} " +
            "and weights defined by {1}").format(
                str(model_file), str(weights_file)
            )
        )
    except IOError as e:
        print(e)
        # create the base pre-trained model
        base_model = application_fn(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        predictions = Dense(7, activation='softmax',
            kernel_regularizer=keras.regularizers.l2(0.01),
            activity_regularizer=keras.regularizers.l1(0.01))(x)

        # this is the model we will train
        model = Model(
            inputs=base_model.input,
            outputs=predictions
        )

        # Save the model as a json
        graph_json = model.to_json()
        model_file = (output_path / "model.json").open('w')
        model_file.write(graph_json)

        print("Created new model pretrained on imagenet at {0}".format(
            str(output_path)
        ))

        epoch = 0

    return model, epoch


def run_training_stage(output_path, epochs, trainable_layers, initial_lr,
                       callbacks, trnds, valds, mdlfn):
    print("BEGINNING NEW TRAINING STAGE")
    model, epoch = get_model(mdlfn, output_path)

    for i, layer in enumerate(model.layers):
        if i not in trainable_layers:
            # print(i, layer, "NO")
            layer.trainable = False
        else:
            # print(i, layer, "YES")
            pass

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=initial_lr),
        loss='categorical_crossentropy', metrics=['categorical_accuracy']
    )
    # model.summary()

    model.fit(
        trnds.make_one_shot_iterator(),
        steps_per_epoch=200,
        initial_epoch=epoch,
        epochs=epoch+epochs,
        validation_data=valds.make_one_shot_iterator(),
        validation_steps=20,
        verbose=1,
        callbacks=callbacks
    )
    return model


def get_callbacks(output_path):
    save_callback = keras.callbacks.ModelCheckpoint(
        str(output_path / "weights.epoch_{epoch:03d}__valacc_{val_categorical_accuracy:.3f}.h5"),
        monitor='val_categorical_accuracy', verbose=1, mode='max',
        save_best_only=True, save_weights_only=True
    )
    tb_callback = keras.callbacks.TensorBoard(log_dir=str(output_path))
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy', factor=0.1, patience=10, verbose=1,
        mode='max', min_delta=0.01
    )
    return [save_callback, tb_callback, lr_callback]


def predict(output_path, X, batch_size=32):
    model, _ = get_model(None, output_path)
    return model.predict(X)

def get_balanced_accuracy(predictions, labels):
    proportions = np.bincount(labels)/labels.shape[0]
    precisions = []
    recalls = []
    for i in range(0, 7):
        tp = np.sum(np.logical_and(
            np.equal(labels, i), np.equal(predictions, i)
        ).astype(np.int32))
        fp = np.sum(np.logical_and(
            np.logical_not(np.equal(labels, i)), np.equal(predictions, i)
        ).astype(np.int32))
        fn = np.sum(np.logical_and(
            np.equal(labels, i), np.logical_not(np.equal(predictions, i))
        ).astype(np.int32))
        precisions = precisions + [tp/(tp+fp+1e-6)]
        recalls = recalls + [tp/(tp+fn+1e-6)]

    print("Precisions:")
    [print("& %.3f " % p, end='') for p in precisions]
    print("\nRecalls")
    [print("& %.3f " % p, end='') for p in recalls]
    print("\nPrecisions: {0}, Recalls: {1}".format(precisions, recalls))
    print("mAP: {0}, mAR:{1}".format(np.mean(precisions), np.mean(recalls)))
    return np.mean(precisions), np.mean(recalls)
