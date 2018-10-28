import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from tensorflow.python.keras.models import load_model

features = np.load("../clean_data/x_test.npy")
labels = np.load("../clean_data/y_test.npy")
model = "../models/Dense.h5"
df = pd.read_csv("../data/bag_data.csv")
class_names = set(df['STATUS'])


def test_acc(fea, lab, mod):
    try:
        neural_nework = load_model(mod)
    except:
        print("model not found")
    prediction = neural_nework.evaluate(x=fea, y=lab, verbose=1, batch_size=64)
    return prediction, neural_nework


def confusion_matrix(feat, labe, neu):
    y_pred_ohe = neu.predict(feat)
    y_pred_labels = np.argmax(y_pred_ohe, axis=1)
    confusion_matrix = metrics.confusion_matrix(y_true=labe.argmax(axis=1), y_pred=y_pred_labels)
    return confusion_matrix


# from sklearn docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


pred, nn = test_acc(features, labels, model)
print("accuracy", pred[1] * 100)
print("loss", pred[0])
conf_mat = confusion_matrix(features, labels, nn)
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
