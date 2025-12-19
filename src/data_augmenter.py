import os
import cv2
import random
import numpy as np
from .constants import RAW_DATA_DIR, AUG_DATA_DIR, IMAGE_SIZE


class DataAugmenter:

    def __init__(self, raw_dir=RAW_DATA_DIR, aug_dir=AUG_DATA_DIR,
                 target_per_class=1500):
        self.raw_dir = raw_dir
        self.aug_dir = aug_dir
        self.target_per_class = target_per_class

    # ---------------- AUGMENT FUNCTIONS -----------------
    def _random_crop(self, img):
        h, w = img.shape[:2]
        crop_ratio = random.uniform(0.6, 0.9)

        ch = int(h * crop_ratio)
        cw = int(w * crop_ratio)

        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)

        crop = img[y:y+ch, x:x+cw]
        return cv2.resize(crop, IMAGE_SIZE)

    def _rotate(self, img):
        angle = random.randint(-20, 20)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))

    def _flip(self, img):
        return cv2.flip(img, 1)

    def _brightness(self, img):
        factor = random.uniform(0.6, 1.4)
        return cv2.convertScaleAbs(img, alpha=factor)

    def _contrast(self, img):
        factor = random.uniform(0.7, 1.6)
        return cv2.convertScaleAbs(img, alpha=factor)

    def _noise(self, img):
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        noisy = img.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _blur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    def _zoom(self, img):
        h, w = img.shape[:2]
        factor = random.uniform(1.0, 1.25)
        new_h, new_w = int(h * factor), int(w * factor)
        resized = cv2.resize(img, (new_w, new_h))
        y = (new_h - h) // 2
        x = (new_w - w) // 2
        return resized[y:y+h, x:x+w]

    # ---------------- APPLY 2–3 RANDOM TRANSFORMS -----------------

    def augment_image(self, img):
        transforms = [
            self._rotate,
            self._flip,
            self._brightness,
            self._contrast,
            self._noise,
            self._blur,
            self._zoom,
        ]

        img_aug = img.copy()
        for t in random.sample(transforms, k=random.randint(2, 3)):
            img_aug = t(img_aug)

        return img_aug



    def augment_dataset(self):
        for class_name in os.listdir(self.raw_dir):

            raw_folder = os.path.join(self.raw_dir, class_name)
            aug_folder = os.path.join(self.aug_dir, class_name)
            os.makedirs(aug_folder, exist_ok=True)

            # Load originals
            originals = []
            for file in os.listdir(raw_folder):
                img = cv2.imread(os.path.join(raw_folder, file))
                if img is not None:
                    originals.append(cv2.resize(img, IMAGE_SIZE))

            original_count = len(originals)
            print(f"[AUG] {class_name}: {original_count} originals")

            needed = self.target_per_class - original_count
            if needed < 0:
                needed = 0

            print(f"[AUG] Need to generate: {needed}")

            for i in range(needed):
                base_img = random.choice(originals)
                aug_img = self.augment_image(base_img)
                cv2.imwrite(os.path.join(aug_folder, f"aug_{i}.jpg"), aug_img)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"  [PROGRESS] Generated {i + 1}/{needed} augmented images...")

            print(f"[AUG] {class_name}: DONE -> {self.target_per_class} images")

        print("\n[SUCCESS] Augmentation completed!")
