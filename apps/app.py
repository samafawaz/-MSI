import cv2
import numpy as np
import joblib
import os
import sys
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.constants import IMAGE_SIZE


MODEL_PATH = "models/svm_model.pkl"

CLASS_NAMES = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash",
    "unknown"
]

# decision logic
CHECKS_MAIN = 7
CHECKS_CARD_DEEP = 13

CONF_GENERAL = 0.45
CONF_SENSITIVE = 0.35

MOVE_THRESHOLD = 15.0
SIMILARITY_MARGIN = 0.08  
CARDBOARD_RECHECK_CLASSES = [0, 1, 3, 4] 



def brighten_like_dataset(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def focus_measure(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_difference(a, b):
    return np.mean(cv2.absdiff(a, b))



class MaterialClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.extractor = FeatureExtractor()
        self.votes = []
        self.last_gray = None

    def reset(self):
        self.votes.clear()
        self.last_gray = None

    def object_moved(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.last_gray is None:
            self.last_gray = gray
            return False
        diff = frame_difference(gray, self.last_gray)
        self.last_gray = gray
        return diff > MOVE_THRESHOLD

    def predict_once(self, roi):
        roi = brighten_like_dataset(roi)
        features = self.extractor.extract_features(roi)
        probs = self.model.predict_proba([features])[0]
        cid = int(np.argmax(probs))
        conf = float(probs[cid])
        return cid, conf, probs

    def majority_vote(self, votes):
        classes = [c for c, _ in votes]
        confs = {}

        for c, f in votes:
            confs.setdefault(c, []).append(f)

        best, count = Counter(classes).most_common(1)[0]
        avg_conf = sum(confs[best]) / len(confs[best])
        return best, avg_conf

    def predict_logic(self, roi):
    
        if self.object_moved(roi):
            self.reset()
            return None, 0.0, len(self.votes)

        cid, conf, probs = self.predict_once(roi)
        if conf >= 0.25:
            self.votes.append((cid, conf))

        if len(self.votes) < CHECKS_MAIN:
            return None, 0.0, len(self.votes)

        best, avg_conf = self.majority_vote(self.votes)

        
        if best == 2:  
            score_sum = {c: 0.0 for c in CARDBOARD_RECHECK_CLASSES}
            count = {c: 0 for c in CARDBOARD_RECHECK_CLASSES}

            for _ in range(CHECKS_CARD_DEEP):
                c, f, _ = self.predict_once(roi)
                if c in score_sum and f >= 0.30:
                    score_sum[c] += f
                    count[c] += 1

            avg_scores = {
                c: score_sum[c] / count[c]
                for c in score_sum if count[c] > 0
            }

            if avg_scores:
                sorted_scores = sorted(
                    avg_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                best_class, best_conf = sorted_scores[0]

                if len(sorted_scores) > 1:
                    second_class, second_conf = sorted_scores[1]
                    if abs(best_conf - second_conf) <= SIMILARITY_MARGIN:
                        return (best_class, second_class), best_conf, CHECKS_MAIN

                return best_class, best_conf, CHECKS_MAIN


        if best in [0, 3] and avg_conf >= CONF_SENSITIVE:
            return best, avg_conf, CHECKS_MAIN

        if avg_conf < CONF_GENERAL:
            return 6, avg_conf, CHECKS_MAIN

        return best, avg_conf, CHECKS_MAIN


# ================= ROI =================
def draw_roi(frame, size=360):
    h, w = frame.shape[:2]
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return x1, y1, x2, y2



def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found")
        return

    clf = MaterialClassifier(MODEL_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=== MATERIAL CLASSIFIER ===")
    print("• 7 main checks")
    print("• Cardboard → deep recheck (13)")
    print("• Similar classes → show both")
    print("• Move object to reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) 
        x1, y1, x2, y2 = draw_roi(frame)
        roi = frame[y1:y2, x1:x2]

        label = "CHECKING..."
        color = (0, 255, 255)

        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if focus_measure(gray) > 60:
                cid, conf, cnt = clf.predict_logic(roi)

                if cnt < CHECKS_MAIN:
                    label = f"CHECKING {cnt}/{CHECKS_MAIN}"
                else:
                    if cid == 6:
                        label = "UNKNOWN"
                        color = (0, 0, 255)
                    elif isinstance(cid, tuple):
                        label = f"{CLASS_NAMES[cid[0]].upper()} / {CLASS_NAMES[cid[1]].upper()}"
                        color = (255, 255, 0)
                    else:
                        label = f"{CLASS_NAMES[cid].upper()} ({conf*100:.1f}%)"
                        color = (0, 255, 0)

        cv2.rectangle(frame, (10, 10), (650, 60), (30, 30, 30), -1)
        cv2.putText(frame, label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        cv2.imshow("Material Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()