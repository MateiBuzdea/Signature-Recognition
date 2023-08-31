import keras
import tensorflow as tf
import numpy as np
from keras import layers
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt


# Define the model
max_pool_size = (2, 2)
conv_kernel_size = (3, 3)
input_shape = (96, 128, 1,)


model = keras.models.Sequential([
    # NN convolutional layers
    keras.layers.Conv2D(64, (7, 7), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D(max_pool_size),
    # keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(max_pool_size),
    # keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(max_pool_size),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    # keras.layers.Dropout(0.2),

    # NN dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    # keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])


# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# Define the callbacks to stop the training if the validation loss does not decrease
class MyStopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.97 and logs.get('val_accuracy') == 1.0 and logs.get('val_loss') < 0.005):
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
callbacks = [learning_rate_reduction]


# ImageDataGenerator custom function to prevent blank image backdoor
def random_invert_img(image, p=0.5):
    return 255 - image if np.random.random() < p else image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=random_invert_img,
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=random_invert_img,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    class_mode='binary',
    target_size=(96, 128),
    batch_size=32,
    color_mode='grayscale',
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    class_mode='binary',
    target_size=(96, 128),
    batch_size=32,
    color_mode='grayscale',
)

test_generator = test_datagen.flow_from_directory(
    'data/validation',
    class_mode='binary',
    target_size=(96, 128),
    batch_size=32,
    color_mode='grayscale',
)

# Edit labels
labels = {'signature_ok': 1, 'signature_wrong': 0}
# train_generator.class_indices = labels
# validation_generator.class_indices = labels
# test_generator.class_indices = labels

# Adjust the weights of the classes to account for the imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}

epochs = 70
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks = callbacks,
)

# Save the model and make predictions
model.save('cnn_model.h5')

predict = model.predict(test_generator)
print(predict)