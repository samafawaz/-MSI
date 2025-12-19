import cv2
import joblib
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.constants import IMAGE_SIZE, CLASS_ID_MAP, UNKNOWN_CLASS_ID
from src.unknown_handler import SVMUnknownHandler

# Map numeric id -> readable name
ID_TO_NAME = {v: k for k, v in CLASS_ID_MAP.items()}
ID_TO_NAME[UNKNOWN_CLASS_ID] = "unknown"


def main():
    print("➡ Loading trained SVM model...")
    svm_pipeline = joblib.load("models/svm_model.pkl")
    print("✔ Model loaded.")

    # Wrap with UNKNOWN handler
    unknown_handler = SVMUnknownHandler(svm_pipeline, threshold=0.65)

    extractor = FeatureExtractor()

    print("➡ Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✔ Webcam ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Resize to the same size used in training
        resized = cv2.resize(frame, IMAGE_SIZE)

        # Extract features
        feat = extractor.extract_features(resized)

        # Predict with UNKNOWN logic
        label_id = unknown_handler.predict_single(feat)
        label_name = ID_TO_NAME.get(label_id, "unknown")

        # Draw label on frame
        cv2.putText(frame, label_name, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("MSI Real-Time Classifier", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("➡ Closing webcam...")
    cap.release()
    cv2.destroyAllWindows()
    print("✔ Bye.")


if __name__ == "__main__":
    main()
