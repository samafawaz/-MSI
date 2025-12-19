
import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

from .feature_extractor import FeatureExtractor
from .constants import CLASS_ID_MAP, IMAGE_SIZE


class KNNClassifier:
    """
    KNN Classifier that works with split data (like SVM)
    Trains on split training data, tests on split test data
    """
    
    def __init__(self):
        # Optimized KNN pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=5,        
                weights="distance",   
                metric="minkowski",
                p=2                   
            ))
        ])
        
        self.feature_extractor = FeatureExtractor()

    def load_data_from_dir(self, data_dir, max_per_class=None):
        """Load images from directory structure"""
        print(f"📂 Loading data from {data_dir}...")
        
        images = []
        labels = []
        
        for class_name, class_id in CLASS_ID_MAP.items():
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  {class_name} directory not found!")
                continue
            
            # Get image files
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit if specified (for KNN we don't need all augmented data)
            if max_per_class and len(files) > max_per_class:
                files = np.random.choice(files, max_per_class, replace=False)
            
            # Load images
            class_images = []
            for filename in files:
                file_path = os.path.join(class_dir, filename)
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    class_images.append(img)
            
            print(f"  {class_name}: {len(class_images)} images")
            
            images.extend(class_images)
            labels.extend([class_id] * len(class_images))
        
        print(f"✅ Data loaded: {len(images)} images")
        return np.array(images), np.array(labels)

    def extract_features(self, images):
        """Extract features from images"""
        print(f"🧠 Extracting features from {len(images)} images...")
        
        features = []
        for i, img in enumerate(images):
            if i % 200 == 0:
                print(f"  Progress: {i}/{len(images)}")
            
            feature_vector = self.feature_extractor.extract_features(img)
            features.append(feature_vector)
        
        features = np.array(features)
        print(f"✅ Features extracted: {features.shape}")
        return features

    def train(self, train_dir="data/split/train", test_dir="data/split/test"):
        """
        Train KNN on split data (same data as SVM)
        
        Args:
            train_dir: Training data directory
            test_dir: Test data directory
        """
        print("🎯 TRAINING KNN ON SPLIT DATA")
        print("Same data as SVM (600 per class)")
        print("=" * 40)
        
        # Load training data (same as SVM - no limit)
        X_train_images, y_train = self.load_data_from_dir(train_dir)
        
        # Load test data (original)
        X_test_images, y_test = self.load_data_from_dir(test_dir)
        
        # Extract features
        X_train_features = self.extract_features(X_train_images)
        X_test_features = self.extract_features(X_test_images)
        
        # Train KNN
        print(f"\n🤖 Training KNN...")
        self.model.fit(X_train_features, y_train)
        
        # Test
        print(f"🎯 Testing...")
        test_predictions = self.model.predict(X_test_features)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n📊 RESULTS:")
        print(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        
        # Detailed report
        class_names = list(CLASS_ID_MAP.keys())
        report = classification_report(y_test, test_predictions, target_names=class_names)
        print(f"\nClassification Report:")
        print(report)
        
        # Save report
        os.makedirs("reports", exist_ok=True)
        with open("reports/knn_report.txt", "w", encoding='utf-8') as f:
            f.write("=== KNN TRAINING REPORT ===\n\n")
            f.write(f"Training samples: {len(X_train_features)} (same as SVM)\n")
            f.write(f"Test samples: {len(X_test_features)} (original)\n")
            f.write(f"Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"✅ Report saved: reports/knn_report.txt")
        return test_accuracy

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

    def save(self, path):
        """Save the trained model"""
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")