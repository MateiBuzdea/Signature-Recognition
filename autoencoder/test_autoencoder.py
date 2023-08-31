import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the Model 
model = keras.models.load_model('./autoencoder_model.h5')

def load_img(path):
    image = tf.keras.utils.load_img(path, target_size=(96, 128), color_mode='grayscale', interpolation='bilinear')
    # plt.imshow(image)
    # plt.show()
    processedImage = tf.keras.utils.img_to_array(image)
    processedImage = np.array([processedImage]) / 255
    return processedImage

path = './test.png'

test_signature = load_img(path)

reconstructed_signature = model.predict(test_signature)

reconstruction_error = tf.keras.losses.mean_squared_error(test_signature.flatten(), reconstructed_signature.flatten())
print(reconstruction_error.numpy())

threshold = 0.005

# Not working..?
if reconstruction_error < threshold:
    print("Access Granted")
else:
    print("Access Denied")