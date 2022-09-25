import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def load_dataset(path: str):
    """
    :return: X_train, y_train, X_valid, y_valid where X.shape = (# clips, 40, 160, 160, 3)
    """
    # training
    videos_train = []
    labels_train = []
    train_path = os.path.join(path, 'train')
    for dir in os.listdir(train_path):
        mod = os.path.join(train_path, dir)  # dataset/train/fights
        for v in tqdm(os.listdir(mod), desc='loading train videos from train/'+mod):
            filename = os.path.join(mod, v)  # dataset/train/fights/vid1.mp4
            videos_train.append(capture_video(filename))
            labels_train.append(1) if dir == 'fights' else labels_train.append(0)

    # validation
    videos_valid = []
    labels_valid = []
    valid_path = os.path.join(path, 'valid')
    for dir in os.listdir(valid_path):
        mod = os.path.join(valid_path, dir)  # dataset/train/fights
        for v in tqdm(os.listdir(mod), desc='loading train videos from train/'+mod):
            filename = os.path.join(mod, v)  # dataset/train/fights/vid1.mp4
            videos_valid.append(capture_video(filename))
            labels_valid.append(1) if mod == 'fights' else labels_train.append(0)

    return np.array(videos_train), tf.keras.utils.to_categorical(labels_train), \
           np.array(videos_valid), tf.keras.utils.to_categorical(labels_valid)


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

