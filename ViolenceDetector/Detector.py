import os
import cv2
from tqdm import tqdm
import matplotlib as plt
import tensorflow as tf
import numpy as np
from .architectures import vg19_lstm, seed_constant
from .utils import capture_video, preprocess_frame, crop_img

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

    def evaluate(self, dataset: str, batch: bool = False):
        """
        self.model.predict(np.expand_dims(X[idx], axis=0))

        Parameters
        ----------
        dataset
            path to dataset with two subfolders 'train' and 'valid', each with two subfolders 'fights' and 'nofights'
            containing the trimmed videos.
        batch
            if True, does model.evaluate, else evaluates one clip per step
        """

        X, y = self.load_dataset(os.path.join(dataset, 'valid'))

        if batch:
            self.model.evaluate(X, y)
        else:
            correct = 0
            for idx in tqdm(range(len(X)), desc='Evaluating'):
                pred = self.model.predict(np.expand_dims(X[idx], axis=0))
                if np.argmax(pred) == np.argmax(y[idx]):
                    correct += 1
            acc = 100*correct/len(X)
            print("Correct predictions: " + str(correct) + " out of " + str(len(X)))
            print("Accuracy (%): " + str(np.round(acc, 2)))

    def forward(self, video, prob_violence: int = 0.95):
        """
        :param prob_violence: probability threshold to consider the video violent
        :param video: array of shape (1, 30, 160, 160, 3)
        :return: (if Violence, probability)
        """
        out = self.model.predict(video)
        if out[0][1] >= prob_violence:
            return True, out[0][1]
        else:
            return False, out[0][1]

    def run_video(self, path: str, stride: int = 2, save: bool = True):
        vid = cv2.VideoCapture(path)
        frame_id = 0
        clip_idx = 0
        seq = []
        clip = np.zeros((self.clip_size, self.image_size, self.image_size, 3), dtype=np.float)
        isViolence = False
        prob = 0

        video_name = os.path.splitext(os.path.basename(path))[0]
        if save:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'MP4V')
            out_video = cv2.VideoWriter('output/'+video_name+'_output.mp4', codec, fps, (width, height))

        while True:
            return_value, frame = vid.read()
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            frame = crop_img(path, frame)
            im_height, im_width, _ = frame.shape

            if frame_id % stride == 0:
                if clip_idx > (self.clip_size-1):
                    # clip is complete for inference
                    isViolence, prob = self.forward(np.expand_dims(clip, axis=0))
                    if isViolence:
                        """cv2.imshow('Violence level: ' + str(np.round(prob, 2)),
                                   np.concatenate((clip[5], clip[15], clip[25], clip[35]), axis=1))
                        cv2.waitKey(0)"""
                        # if violence fight1 clip
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        vio = cv2.VideoWriter("./output/" + video_name + '_' + str(frame_id) + ".avi", fourcc, 10.0, (im_width, im_height))
                        # vio = cv2.VideoWriter("./videos/output-"+str(j)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (300, 400))
                        for frameinss in seq:
                            vio.write(frameinss)
                        vio.release()
                    clip_idx = 0
                    clip = np.zeros((self.clip_size, self.image_size, self.image_size, 3), dtype=np.float)
                    seq = []
                else:
                    seq.append(frame)
                    clip[clip_idx, :, :, :] = preprocess_frame(frame, self.image_size)
                    clip_idx += 1

            frame_id += 1

            mess = 'Violence!' if isViolence else 'No Violence'
            color = (0, 0, 255) if isViolence else (0, 255, 0)
            result = cv2.resize(frame, (int(im_width*0.6), int(im_height*0.6)))
            result = cv2.putText(result, str(frame_id)+': '+mess+' ('+str(np.round(prob, 3))+')', (int(im_width * 0.05), int(im_height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, lineType=cv2.LINE_AA)
            cv2.imshow('', result)
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if save:
                out_video.write(result)

        vid.release()
        cv2.destroyAllWindows()
