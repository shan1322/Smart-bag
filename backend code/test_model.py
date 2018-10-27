import numpy as np
from tensorflow.python.keras.models import load_model

features = np.load("../clean_data/x_test.npy")
labels = np.load("../clean_data/y_test.npy")
model = "../models/Dense.h5"


def test_acc(fea, lab, mod):
    neural_nework = load_model(mod)
    prediction = neural_nework.evaluate(x=features, y=labels, verbose=1, batch_size=64)
    return prediction


pred = test_acc(features, labels, model)
print("accuracy", pred[1] * 100)
print("loss", pred[0])
