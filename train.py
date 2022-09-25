from ViolenceDetector.Detector import Detector

if __name__ == '__main__':
    dataset = r'data/CustomViolenceDataset'
    detector = Detector(dataset)
    detector.train()
