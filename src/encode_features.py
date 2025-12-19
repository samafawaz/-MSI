import os
import numpy as np
import cv2
from .data_loader import DatasetLoader
from .feature_extractor import FeatureExtractor
from .constants import RAW_DATA_DIR, SPLIT_DATA_DIR, FEATURES_DIR, IMAGE_SIZE

class FeatureEncoder:
    """
    Converts images from SPLIT dataset into numerical feature vectors.
    
    IMPORTANT: Uses data/split/train/ and data/split/test/ to prevent data leakage.
    This ensures augmented training images and original test images are properly separated.
    """

    def __init__(self, train_dir=None, test_dir=None):
        """
        Initialize FeatureEncoder with split data directories.
        
        Args:
            train_dir: Training data directory (default: data/split/train)
            test_dir: Test data directory (default: data/split/test)
        """
        if train_dir is None:
            train_dir = os.path.join(SPLIT_DATA_DIR, "train")
        if test_dir is None:
            test_dir = os.path.join(SPLIT_DATA_DIR, "test")
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.extractor = FeatureExtractor()

    def encode_and_save(self, save_separate=True):
        """
        Extract features from split dataset and save.
        
        Args:
            save_separate: If True, saves train and test features separately
        """
        print("[INFO] Loading TRAINING dataset from split...")
        train_loader = DatasetLoader(data_dir=self.train_dir)
        train_images, train_labels = train_loader.load_dataset()
        print(f"[OK] Loaded {len(train_images)} training images")

        print("[INFO] Loading TEST dataset from split...")
        test_loader = DatasetLoader(data_dir=self.test_dir)
        test_images, test_labels = test_loader.load_dataset()
        print(f"[OK] Loaded {len(test_images)} test images")

        print("[INFO] Extracting features (this may take some time)...")
        
        # Extract training features
        X_train = []
        y_train = []
        for i, (img, label) in enumerate(zip(train_images, train_labels)):
            if i % 200 == 0:
                print(f"  [PROGRESS] Training: {i}/{len(train_images)}")
            features = self.extractor.extract_features(img)
            X_train.append(features)
            y_train.append(label)

        # Extract test features
        X_test = []
        y_test = []
        for i, (img, label) in enumerate(zip(test_images, test_labels)):
            if i % 200 == 0:
                print(f"  [PROGRESS] Test: {i}/{len(test_images)}")
            features = self.extractor.extract_features(img)
            X_test.append(features)
            y_test.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print("[INFO] Saving feature vectors...")
        if not os.path.exists(FEATURES_DIR):
            os.makedirs(FEATURES_DIR)

        if save_separate:
            # Save train and test separately (prevents leakage)
            np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
            np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
            np.save(os.path.join(FEATURES_DIR, "X_test.npy"), X_test)
            np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)
            print("[SUCCESS] Feature extraction complete!")
            print(f"[INFO] Training: X={X_train.shape}, y={y_train.shape}")
            print(f"[INFO] Test: X={X_test.shape}, y={y_test.shape}")
        else:
            # Legacy: combine for backward compatibility (use with caution)
            X = np.concatenate([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            np.save(os.path.join(FEATURES_DIR, "X.npy"), X)
            np.save(os.path.join(FEATURES_DIR, "y.npy"), y)
            print("[SUCCESS] Feature extraction complete!")
            print(f"[WARN] Combined dataset: X={X.shape}, y={y.shape}")
            print("[WARN] Ensure you don't split this combined dataset again!")
