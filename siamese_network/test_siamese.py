import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K


def initialize_weights(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

# Load the Model 
# model = keras.models.load_model('./siamese_model.h5', custom_objects={'initialize_weights': initialize_weights, 'initialize_bias': initialize_bias})
model = keras.models.load_model('./siamese_model.h5')
model.trainable = False

def load_img(path):
    image = tf.keras.utils.load_img(
        path,
        target_size=(96, 128),
        color_mode='grayscale',
        interpolation='bilinear',
    )
    # plt.imshow(image)
    # plt.show()
    processedImage = tf.keras.utils.img_to_array(image)
    processedImage = np.array([processedImage]) / 255
    return processedImage

orig = load_img('./orig.jpg')
copy = load_img('./test.png')


p = model.predict([orig, copy])[0][0]
print("{:.20f}".format(p))
status = round(p)


if status == 1:
    print("Access Granted")
else:
    print("Access Denied")