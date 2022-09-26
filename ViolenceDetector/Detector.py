import matplotlib as plt
from .architectures import *
from .utils import *


class Detector:
    def __init__(self, dataset: str, input_size: int = 40):
        self.input_size = input_size
        self.model = vg19_lstm(input_size)
        self.dataset_path = dataset

    def train(self, lr: float = 0.0005, epochs: int = 10, batch_size: int = 1, plot: bool = True):

        optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                         min_delta=1e-5, verbose=0,
                                                         mode='min', restore_best_weights=True)
        mcp_save = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', save_best_only=True,
                                                      monitor='val_loss', mode='min')
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1,
                                                              verbose=2, factor=0.5, min_lr=0.0000001)

        clips_train, labels_train, clips_valid, labels_valid = load_dataset(self.dataset_path)

        train_hist = self.model.fit(x=clips_train, y=labels_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                    verbose=1,
                                    validation_data=(clips_valid, labels_valid))
        self.model.evaluate(clips_valid, labels_valid)

        if plot:
            epochs = range(len(train_hist.history['loss']))
            plt.plot(epochs, train_hist.history['loss'], label='loss')
            plt.plot(epochs, train_hist.history['val_loss'], label='val_loss')
            plt.plot(epochs, train_hist.history['accuracy'], label='accuracy')
            plt.plot(epochs, train_hist.history['val_accuracy'], label='val_accuracy')
            plt.legend()
            plt.show()

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
