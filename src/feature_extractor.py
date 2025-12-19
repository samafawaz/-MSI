import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from .constants import IMAGE_SIZE


class FeatureExtractor:
    """
    Final optimized feature extractor for high-accuracy SVM (85%+).
    Uses:
      - HOG descriptor (main powerful feature)
      - LBP texture histogram
      - HSV color histogram
    """

    def __init__(self):
        
        self.hog_params = dict(
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

    
    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, **self.hog_params)
        return features

 
    def extract_lbp(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Uniform LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

        # 59-bin histogram
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist  # 59 dims

 
    def extract_hsv_hist(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist  # 512 dims

  
    def extract_features(self, image):
        image = cv2.resize(image, IMAGE_SIZE)

        hog_vec = self.extract_hog(image)
        lbp_vec = self.extract_lbp(image)
        hsv_vec = self.extract_hsv_hist(image)


       
        return np.concatenate([hog_vec, lbp_vec, hsv_vec])