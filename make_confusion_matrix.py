import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    uncm = cm.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(uncm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

preds = np.argmax(np.load("ensemble_validation_predictions.npy"), axis=-1)
labels = np.load("validation_y.npy")
confusion = np.zeros((7,7), np.int32)

for i in range(labels.shape[0]):
    confusion[labels[i], preds[i]] = confusion[labels[i], preds[i]] + 1

plt.figure()
plot_confusion_matrix(
    confusion,
    classes=[
        "Melanoma", "Melanocytic\nNevus", "Basal\nCell\nCarcinoma",
        "Actinic\nKeratosis", "Benign\nKeratosis", "Dermatofibroma",
        "Vascular\nLesion"
    ],
    title="Unnormalized Confusion Matrix"
)

plt.figure()
plot_confusion_matrix(
    confusion,
    classes=[
        "Melanoma", "Melanocytic\nNevus", "Basal\nCell\nCarcinoma",
        "Actinic\nKeratosis", "Benign\nKeratosis", "Dermatofibroma",
        "Vascular\nLesion"
    ],
    title="Normalized Confusion Matrix",
    normalize=True
)
plt.show()
