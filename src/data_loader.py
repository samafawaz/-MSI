import os
import cv2
from typing import List, Tuple
from .constants import CLASS_ID_MAP, RAW_DATA_DIR, IMAGE_SIZE

class DatasetLoader:
    """
    Loads the raw dataset from folder structure:
    data/raw/class_name/*.jpg
    """

    def __init__(self, data_dir: str = RAW_DATA_DIR):
        self.data_dir = data_dir

    def load_dataset(self) -> Tuple[List, List]:
        """
        Loads all images and labels from raw dataset.
        Returns:
            images: list of image arrays
            labels: list of numeric labels
        """
        images = []
        labels = []

        for class_name, class_id in CLASS_ID_MAP.items():
            class_folder = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_folder):
                print(f"[WARNING] Missing folder: {class_folder}")
                continue

            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)

                # Read image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"[ERROR] Cannot read: {file_path}")
                    continue

                # Resize
                img = cv2.resize(img, IMAGE_SIZE)

                images.append(img)
                labels.append(class_id)

        return images, labels
