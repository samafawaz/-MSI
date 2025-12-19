import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_svm import SVMClassifier

print("[INFO] RUNNING SVM TRAINING")
print("Training on split data")
print("=" * 40)

svm = SVMClassifier()
accuracy = svm.train()

svm.save("models/svm_model.pkl")

print(f"\n[SUCCESS] SVM training completed!")
print(f"[INFO] Accuracy: {accuracy*100:.1f}%")
print(f"[INFO] Model saved: models/svm_model.pkl")
