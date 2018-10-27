import tensorflow as tf

converter = tf.contrib.lite.TocoConverter.from_keras_model_file("../models/Dense.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
