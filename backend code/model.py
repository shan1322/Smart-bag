import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential

features = np.load("../clean_data/x_train.npy")
labels = np.load("../clean_data/y_train.npy")


def dense_net(data):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(data.shape[1],)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = dense_net(features)
model.fit(x=features, y=labels, batch_size=64, epochs=10, verbose=1)
model.save("../models/Dense.h5")
