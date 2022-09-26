import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def capture_video(filename: str, clip_size: int = 40, image_size: int = 160):
    vid = cv2.VideoCapture(filename)

    clip = np.zeros((clip_size, image_size, image_size, 3), dtype=np.float)

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    temporal_stride = max(int(total_frames / clip_size), 1)

    for count in range(clip_size):
        # move video to desired frame given stride
        vid.set(cv2.CAP_PROP_POS_FRAMES, count * temporal_stride)

        # read next frame
        success, frame = vid.read()
        if not success: break
        clip[count, :, :, :] = preprocess_frame(frame, image_size)

    return clip


def preprocess_frame(frame, image_size: int):
    frm = cv2.resize(frame, (image_size, image_size))
    frm = frm / 255.
    # frm = np.expand_dims(frm, axis=0)
    return frm


def crop_img(video: str, frame):
    if 'fight1' in video:
        return frame[100:600, 500:1000, :]
    elif 'fight2' in video:
        # frame[110:, 350:, :]
        return frame[110:600, 700:, :]
    elif any(word in video for word in ['fight3', 'fight4', 'fight5', 'fight6']):
        # frame[110:, 700:1880, :]
        return frame[:, 300:, :]
    else:
        return frame


def clear_folder(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class BatchTraining(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size, clip_size, image_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.image_size = image_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = np.array([capture_video(filename=x, clip_size=self.clip_size, image_size=self.image_size)
                            for x in tqdm(batch_x, desc='Preprocessing')])
        return batch_x, batch_y

