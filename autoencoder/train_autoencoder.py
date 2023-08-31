import keras
from keras import layers, models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import os


# Define the model
max_pool_size = (2, 2)
conv_kernel_size = (3, 3)
input_shape = (96, 128, 1,)

model = models.Sequential()

# Encoder
model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.BatchNormalization())

# Decoder
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))


# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
model.summary()

class MyStopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.97 and logs.get('val_accuracy') == 1.0 and logs.get('val_loss') < 0.01):
            print("\nReached target accuracy so cancelling training!")
            self.model.stop_training = True
my_earlystop = MyStopCallback()

learning_rate_reduction = ReduceLROnPlateau(
    monitor='accuracy', 
    patience=2, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001,
)
callbacks = [learning_rate_reduction]


def process_image(path):
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

def load_images(dir='siamese_data\\Train'):
    '''
    dir: directory where the images are located
    samples: number of samples to load
    returns: classes and images
    images: np array of shape (num_classes, samples, height, width, channels)
    '''
    images = np.array([])

    for subdir, dirs, files in os.walk(dir):
        for file in files:
            img = process_image(os.path.join(subdir, file))

            if images.size == 0:
                images = img
            else:
                images = np.concatenate((images, img), axis=0)

    return images


def generate(batch_size=32):
    '''
    a generator that returns a batch of pairs and labels
    '''
    while True:
        images = np.random.shuffle(load_images(batch_size))
        yield images[:batch_size]


img = load_images()

epochs = 50

history = model.fit(
    img, img,
    epochs=epochs,
    batch_size=32,
)

# Save the model and make predictions
model.save('autoencoder_model.h5')

# predict = model.predict(test_generator)
# print(predict)