import tensorflow as tf
import numpy as np
import random

seed_constant = 1234
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def vg19_lstm():
    """
    Official Implementation of 'Robust Real-Time Violence Detection in Video Using CNN And LSTM'
    # Read sequence of frames in 4d tensor (frame, H, W, RGB)
    # Apply pre-trained CNN for each frame
    # Group the result from the previous step and flatten the tensor to be a 2d shape (frames, SP) where SP is (H*W*RGB)
      and represent a spatial feature vector for one frame.
    # Use the previous step output as feature vector input to LSTM where SP represent input and Frame represent
      time step ex for 30 frame input we have (SP1, SP2 ..SP30) each goes in a time step of LSTM.
    # Take full sequence prediction from LSTM and feed it to a dense layer in a time distributed manner
    # Take the global average of the previous step output to get the result as a 1d tensor.
    # Feed the output of the previous step into the output layer (dense layer with sigmoid activation which represents
      the probability of violence existence in the given video)
    """
    vg19 = tf.keras.applications.VGG19
    base_model = vg19(include_top=False, weights='imagenet', input_shape=(160, 160, 3))

    # Freeze the layers except the last 4 layers
    for layer in base_model.layers:
        layer.trainable = False

    cnn = tf.keras.models.Sequential()
    cnn.add(base_model)
    cnn.add(tf.keras.layers.Flatten())

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)))

    model.add(tf.keras.layers.LSTM(30, return_sequences=True))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(90)))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(2, activation="sigmoid"))
    print(model.summary())
    return model