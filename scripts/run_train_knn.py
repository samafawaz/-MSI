import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_knn import KNNClassifier

print("[INFO] RUNNING KNN TRAINING")
print("Training on split data (same as SVM)")
print("=" * 40)

knn = KNNClassifier()
accuracy = knn.train()

knn.save("models/knn_model.pkl")

print(f"\n[SUCCESS] KNN training completed!")
print(f"[INFO] Accuracy: {accuracy*100:.1f}%")
print(f"[INFO] Model saved: models/knn_model.pkl")