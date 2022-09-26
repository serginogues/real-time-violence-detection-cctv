from ViolenceDetector.Detector import Detector

CLIP_SIZE = 40
IMAGE_SIZE = 160
WEIGHTS = 'checkpoint.hdf5'

TRAIN = False
LR = 0.0005
EPOCHS = 10
BATCH = 1

DATASET = r'data/CustomViolenceDataset'

if __name__ == '__main__':
    weights = None if TRAIN else WEIGHTS
    detector = Detector(weights=weights, clip_size=CLIP_SIZE, image_size=IMAGE_SIZE, learning_rate=LR)
    detector.train(dataset=DATASET, epochs=EPOCHS, batch_size=BATCH) if TRAIN else detector.evaluate(dataset=DATASET)


