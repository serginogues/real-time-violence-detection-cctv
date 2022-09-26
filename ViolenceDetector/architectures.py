import tensorflow as tf
import numpy as np
import random

seed_constant = 1234
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def vg19_lstm(weights=None, clip_size: int = 40, image_size: int = 160, lr: float = 0.0005):
    """
    Implementation of 'Robust Real-Time Violence Detection in Video Using CNN And LSTM'

    Parameters
    ----------
    weights
        path to pretrained weights .h5. If not None, weights are loaded
    clip_size
        size of the input frame batch
    image_size
        resize during preprocessing
    lr
        learning rate
    """
    vg19 = tf.keras.applications.VGG19
    base_model = vg19(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))

    if weights is None:
        # Freeze the layers except the last 4 layers
        for layer in base_model.layers:
            layer.trainable = False

    cnn = tf.keras.models.Sequential()
    cnn.add(base_model)
    cnn.add(tf.keras.layers.Flatten())

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.TimeDistributed(cnn, input_shape=(clip_size, image_size, image_size, 3)))

    model.add(tf.keras.layers.LSTM(clip_size, return_sequences=True))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(90)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
    print(model.summary())

    if weights is not None:
        model.load_weights(weights)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    return model
