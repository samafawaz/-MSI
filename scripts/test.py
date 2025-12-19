import cv2
import joblib
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.constants import IMAGE_SIZE

# MUST match training order
CLASS_NAMES = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash",
]

def preprocess_like_dataset(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def test_image(path, model, extractor):
    print("\n➡ Testing:", path)

    img = cv2.imread(path)
    if img is None:
        print("❌ Cannot read image")
        return

    img = preprocess_like_dataset(img)
    features = extractor.extract_features(img)

    probs = model.predict_proba([features])[0]
    cid = int(np.argmax(probs))
    conf = float(probs[cid])

    print(f"✔ Prediction: {CLASS_NAMES[cid].upper()}")
    print(f"✔ Confidence: {conf:.3f}")

    print("— All class probabilities —")
    for i, p in enumerate(probs):
        print(f"  {CLASS_NAMES[i]:10s}: {p:.3f}")


if __name__ == "__main__":

    model = joblib.load("models/svm_model.pkl")
    extractor = FeatureExtractor()

    TEST_PATHS = [
        r"D:\Downloads\dataset\cardboard\efb2516b-eefd-4e59-a7aa-9470b9c7e77c.jpg",
        r"D:\Downloads\dataset\glass\d25b7067-d140-47e5-9138-5be697b192b8.jpg",
        r"D:\Downloads\dataset\metal\c34dd3ba-0963-4221-adb4-66dedf1d02d2.jpg",
        r"D:\Downloads\dataset\paper\d541ee61-9ea9-4c30-84b3-13c2d325210b.jpg",
        r"D:\Downloads\dataset\plastic\d9f3c3e9-73a8-4584-8891-dbde0ab3d764.jpg",
        r"D:\Downloads\dataset\trash\d3fcfa27-7fd9-4e7a-b615-75fd0c6d3ec6.jpg",
    ]

    for p in TEST_PATHS:
        test_image(p, model, extractor)
