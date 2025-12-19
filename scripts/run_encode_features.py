import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encode_features import FeatureEncoder

if __name__ == "__main__":
    print("[INFO] RUNNING FEATURE ENCODING")
    print("Using split data to prevent data leakage")
    print("=" * 50)
    
    # Use split data directories (prevents leakage)
    encoder = FeatureEncoder()
    encoder.encode_and_save(save_separate=True)
    
    print("\n[SUCCESS] Feature encoding completed!")
    print("[INFO] Features saved to: data/features/")
    print("[INFO] Files: X_train.npy, y_train.npy, X_test.npy, y_test.npy")
