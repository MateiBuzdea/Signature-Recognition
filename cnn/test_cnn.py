import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = './test.png'

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

# Load the Model 
model = keras.models.load_model('./cnn_model.h5')

p = model.predict(processedImage)
print(p)
status = round(p[0][0])

if status == 0:
    print("Access Granted")
else:
    print("Access Denied")