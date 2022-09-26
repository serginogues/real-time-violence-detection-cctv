import cv2
import numpy as np


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

