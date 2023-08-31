import keras
import signal
from functools import partial
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras import layers, models
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, Adagrad
import keras.backend as K


# When Ctrl+C is pressed, stop the training and save the model
def ctrlc_handler(model, sig, frame):
    print('You pressed Ctrl+C!')
    model.save('./siamese_model_2.h5')
    exit(0)


# Define the model
max_pool_size = (2, 2)
conv_kernel_size = (3, 3)
input_shape = (96, 128, 1,)


# As taken from https://github.dev/akshaysharma096/Siamese-Networks/
# The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
# suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
def initialize_weights(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


signature_left = layers.Input(shape=input_shape)
signature_right = layers.Input(shape=input_shape)
args = {
    # 'kernel_initializer': initialize_weights,
    # 'bias_initializer': initialize_bias,
    'kernel_regularizer': keras.regularizers.l2(2e-4)
}

def create_siamese_network(input_shape):
    input_signature = layers.Input(shape=input_shape)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (8, 8), activation='relu', input_shape=input_shape, **args))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', **args))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', **args))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', **args))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='sigmoid', **args))

    output_signature = model(input_signature)
    signature_network = models.Model(input_signature, output_signature, name='signature_network')
    model.summary()

    output_left = signature_network(signature_left)
    output_right = signature_network(signature_right)

    L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    distance = L1_layer([output_left, output_right])

    siamese_model = models.Model(
        inputs=[signature_left, signature_right],
        outputs=distance,
        name='siamese_network',
    )
    siamese_model.summary()

    return siamese_model


def create_dense_layers_model(input_shape):
    input_diff = layers.Input(shape=input_shape)

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    # model.add(layers.Dense(256, activation='relu', **args))
    model.add(layers.Dense(1, activation='sigmoid')) #, bias_initializer=initialize_bias))

    output_diff = model(input_diff)
    return models.Model(
        input_diff,
        output_diff,
        name='dense_layers_network',
    )


# Create the model
siamese_model = create_siamese_network(input_shape)
dense_layers_model = create_dense_layers_model(siamese_model.output_shape[1:])
dense_layers_model.summary()

final_output = dense_layers_model(siamese_model.output)
final_model = models.Model(
    inputs=[signature_left, signature_right],
    outputs=final_output,
    name='final_model',
)
final_model.summary()


# Compile the model
# optimizer = Adagrad(learning_rate=0.0001)
optimizer = Adam(learning_rate=0.0001)
final_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Define the callbacks to stop the training if the validation loss does not decrease
class MyStopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.97 and logs.get('val_accuracy') > 0.97 and logs.get('val_loss') < 0.08):
            print("\nReached target accuracy so cancelling training!")
            self.model.stop_training = True
my_earlystop = MyStopCallback()

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=2, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001,
    mode='min',
)
callbacks = [learning_rate_reduction, my_earlystop]


# Create the generators for the siamese network
def load_images(dir='siamese_data\\Train', samples=30):
    '''
    dir: directory where the images are located
    samples: number of samples to load
    returns: classes and images
    images: np array of shape (num_classes, samples, height, width, channels)
    '''
    classes = np.array(os.listdir(dir))
    classes = [int(c) for c in classes if c.isdigit()]
    images = np.array([])

    for subdir, dirs, files in os.walk(dir):
        if subdir == dir:
            continue
        
        # get a number of random samples from each class
        # save them into a np array for the generator
        for i in range(samples):
            file = np.random.choice(files)
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(subdir, file),
                target_size=input_shape,
                color_mode='grayscale',
                interpolation='bilinear',
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            if i == 0:
                signature_images = img
            else:
                signature_images = np.concatenate((signature_images, img), axis=0)
        signature_images = np.expand_dims(signature_images, axis=0)

        if images.size == 0:
            images = signature_images
        else:
            images = np.concatenate((images, signature_images), axis=0)

    return classes, images

classes, images = load_images()


def get_batch(batch_size=32):
    '''
    the batch will have half same class images and half different class images
    batch_size: number of pairs to return
    subset: 'train' or 'test'
    returns: pairs and labels
    '''

    pairs = [np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2])) for i in range(2)]
    labels = np.zeros((batch_size,))

    for i in range(batch_size):
        cls_1 = cls_2 = np.random.choice(classes)
        while i >= batch_size // 2 and cls_1 == cls_2:
            cls_2 = np.random.choice(classes)

        idx_1 = np.random.randint(0, images.shape[1])
        idx_2 = np.random.randint(0, images.shape[1])

        # generate a pair of images
        pairs[0][i, :, :, :] = images[cls_1, idx_1, :, :, :]
        pairs[1][i, :, :, :] = images[cls_2, idx_2, :, :, :]
        labels[i] = 1 if cls_1 == cls_2 else 0

    return pairs, labels


def generate(batch_size=32):
    '''
    a generator that returns a batch of pairs and labels
    '''
    while True:
        pairs, labels = get_batch(batch_size)
        yield pairs, labels


# Train the model
signal.signal(signal.SIGINT, partial(ctrlc_handler, final_model))

epochs = 50
history = final_model.fit(
    generate(batch_size=32),
    steps_per_epoch=100,
    epochs=epochs,
    validation_data=generate(batch_size=32),
    validation_steps=50,
    callbacks=callbacks,
)

# Save the model
print('Saving the model...')
final_model.save('siamese_model.h5')
