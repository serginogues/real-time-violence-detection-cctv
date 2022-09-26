import os
from tqdm import tqdm
import matplotlib as plt
import tensorflow as tf
import numpy as np
from .architectures import vg19_lstm, seed_constant
from .utils import capture_video

tf.random.set_seed(seed_constant)


class Detector:
    def __init__(self, weights=None, clip_size: int = 40, image_size: int = 160, learning_rate: float = 0.0005):
        """
        If weights are provided, then they are loaded and the model is ready for inference
        , otherwise the model will be ready for training.
        """
        self.lr = learning_rate
        self.clip_size = clip_size
        self.image_size = image_size
        self.model = vg19_lstm(weights, clip_size, image_size, learning_rate)

    def load_dataset(self, path: str):
        x = []
        y = []
        for dir in os.listdir(path):
            clase = os.path.join(path, dir)  # dataset/train/fights
            for v in tqdm(os.listdir(clase), desc='loading  videos from ' + clase):
                filename = os.path.join(clase, v)  # dataset/train/fights/vid1.mp4
                x.append(capture_video(filename=filename, clip_size=self.clip_size, image_size=self.image_size))
                y.append(1) if dir == 'fights' else y.append(0)
        return np.array(x), tf.keras.utils.to_categorical(y)

    def train(self, dataset: str,
              epochs: int = 10, batch_size: int = 1,
              plot: bool = True):

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                         min_delta=1e-5, verbose=0,
                                                         mode='min', restore_best_weights=True)
        mcp_save = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', save_best_only=True,
                                                      monitor='val_loss', mode='min')
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1,
                                                              verbose=2, factor=0.5, min_lr=0.0000001)

        clips_train, labels_train = self.load_dataset(os.path.join(dataset, 'train'))
        clips_valid, labels_valid = self.load_dataset(os.path.join(dataset, 'valid'))

        train_hist = self.model.fit(x=clips_train, y=labels_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                    verbose=1,
                                    validation_data=(clips_valid, labels_valid))

        print("End training")

        if plot:
            epochs = range(len(train_hist.history['loss']))
            plt.plot(epochs, train_hist.history['loss'], label='loss')
            plt.plot(epochs, train_hist.history['val_loss'], label='val_loss')
            plt.plot(epochs, train_hist.history['accuracy'], label='accuracy')
            plt.plot(epochs, train_hist.history['val_accuracy'], label='val_accuracy')
            plt.legend()
            plt.show()

    def evaluate(self, dataset: str):
        """
        self.model.predict(np.expand_dims(X[idx], axis=0))

        Parameters
        ----------
        dataset
            path to dataset with two subfolders 'train' and 'valid', each with two subfolders 'fights' and 'nofights'
            containing the trimmed videos.
        """

        X, y = self.load_dataset(os.path.join(dataset, 'valid'))
        self.model.evaluate(X, y)

    def predict(self, video, prob_violence: int = 0.95):
        """
        :param prob_violence: probability threshold to consider the video violent
        :param video: array of shape (1, 30, 160, 160, 3)
        :return: (True/False, prob)
        """
        out = self.model(video)
        if out[0][1] >= prob_violence:
            return True, out[0][1]
        else:
            return False, out[0][1]

    def run_video(self, path: str, save: bool):
        pass
