import os
import cv2
import random
import shutil
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import RAW_DATA_DIR, SPLIT_DATA_DIR, IMAGE_SIZE


class TrainTestSplitter:
    def __init__(
        self,
        raw_dir=RAW_DATA_DIR,
        split_dir=SPLIT_DATA_DIR,
        train_ratio=0.7,
        target_train_per_class=1500
    ):
        self.raw_dir = raw_dir
        self.split_dir = split_dir
        self.train_ratio = train_ratio
        self.target_train_per_class = target_train_per_class

        self.train_dir = os.path.join(split_dir, "train")
        self.test_dir = os.path.join(split_dir, "test")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    # ---------------- BASIC AUGMENTATIONS ----------------
    def _rotate(self, img):
        angle = random.randint(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def _brightness(self, img):
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-15, 15)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _noise(self, img):
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def augment(self, img):
        if random.random() < 0.5:
            img = self._rotate(img)
        if random.random() < 0.5:
            img = self._brightness(img)
        if random.random() < 0.3:
            img = self._noise(img)
        return img

    # ---------------- MAIN PROCESS ----------------
    def run(self):
        for class_name in os.listdir(self.raw_dir):

            class_path = os.path.join(self.raw_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            images = []
            for f in os.listdir(class_path):
                img = cv2.imread(os.path.join(class_path, f))
                if img is not None:
                    images.append(cv2.resize(img, IMAGE_SIZE))

            random.shuffle(images)

            split_idx = int(len(images) * self.train_ratio)
            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]

            # create folders
            train_class_dir = os.path.join(self.train_dir, class_name)
            test_class_dir = os.path.join(self.test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # ---------- SAVE TEST (NO AUGMENTATION) ----------
            for i, img in enumerate(test_imgs):
                cv2.imwrite(
                    os.path.join(test_class_dir, f"test_{i}.jpg"),
                    img
                )

            # ---------- SAVE TRAIN ORIGINAL ----------
            for i, img in enumerate(train_imgs):
                cv2.imwrite(
                    os.path.join(train_class_dir, f"train_{i}.jpg"),
                    img
                )

            # ---------- AUGMENT TRAIN ONLY ----------
            needed = self.target_train_per_class - len(train_imgs)

            aug_i = 0
            while needed > 0:
                base = random.choice(train_imgs)
                aug = self.augment(base)
                cv2.imwrite(
                    os.path.join(train_class_dir, f"aug_{aug_i}.jpg"),
                    aug
                )
                aug_i += 1
                needed -= 1

            print(
                f"[SPLIT] {class_name}: "
                f"train={len(train_imgs)} → {self.target_train_per_class}, "
                f"test={len(test_imgs)}"
            )

        print("\n✔ Train/Test split completed correctly!")


if __name__ == "__main__":
    print("🎯 RUNNING DATA SPLITTER")
    print("80% Training + 20% Testing (optimized)")
    print("=" * 40)
    
    # Create splitter (optimized for higher accuracy)
    splitter = TrainTestSplitter(
        train_ratio=0.8,
        target_train_per_class=800  # 800 images per class for optimal accuracy
    )
    
    # Run the split
    splitter.run()
    
    print("\n✅ Data splitting completed!")
    print("📁 Results:")
    print("  - Training data: data/split/train/")
    print("  - Test data: data/split/test/")