import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

path = "../data/bag_data.csv"


def load_file(loc):
    df = pd.read_csv(loc)
    sta = df.iloc[:, 13].values
    un_processed_features = df.iloc[:, :13]
    return sta, un_processed_features


def label_encoder(variable):
    le = preprocessing.LabelEncoder()
    encoded = le.fit(variable)
    numbers = encoded.transform(variable)
    one_hot = np_utils.to_categorical(numbers, 5)
    return np.array(one_hot)


def train_test(feature, label):
    x1, x2, y1, y2 = train_test_split(feature, label, test_size=0.1, random_state=42)
    return x1, x2, y1, y2


raw_labels, raw_features = load_file(path)
y = label_encoder(raw_labels)
x_train, x_test, y_train, y_test = train_test_split(raw_features, y)

np.save("../clean_data/x_train.npy", x_train)
np.save("../clean_data/x_test.npy", x_test)
np.save("../clean_data/y_train.npy", y_train)
np.save("../clean_data/y_test.npy", y_test)
