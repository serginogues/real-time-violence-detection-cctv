import os
from ViolenceDetector.Detector import Detector
from config import *

PATH = r'data\ToProcess'
STRIDE = 2

if __name__ == '__main__':
    ll = [os.path.join(PATH, x) for x in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, x))]
    detector = Detector(weights=WEIGHTS, clip_size=CLIP_SIZE, image_size=IMAGE_SIZE, learning_rate=LR)

    for x in ll:
        detector.run_video(x, STRIDE)
