## Import Libraries
import tensorflow as tf
from tensorflow import keras


## Convert downloaded Keras model to TFlite model

model = keras.models.load_model('cerv_fracture_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("cerv_fracture_model.tflite", "wb") as f_out:
    f_out.write(tflite_model)