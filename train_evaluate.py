from ViolenceDetector.Detector import Detector
from config import *

if __name__ == '__main__':
    weights = None if TRAIN else WEIGHTS
    detector = Detector(weights=weights, clip_size=CLIP_SIZE, image_size=IMAGE_SIZE, learning_rate=LR)
    detector.train(dataset=DATASET, epochs=EPOCHS, batch_size=BATCH) if TRAIN else detector.evaluate(dataset=DATASET)


