import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmenter import DataAugmenter

if __name__ == "__main__":
    print("[WARN] =========================================")
    print("[WARN] DEPRECATED: This script may cause data leakage!")
    print("[WARN] =========================================")
    print("[INFO] RECOMMENDED: Use 'python scripts/run_data_splitter.py' instead")
    print("[INFO] That script properly splits FIRST, then augments training only.")
    print("")
    print("[INFO] If you still want to use this script:")
    print("[INFO] - Only use it if you understand data leakage risks")
    print("[INFO] - Do NOT combine raw + augmented data before splitting")
    print("=" * 50)
    
    response = input("Continue anyway? (yes/no): ").strip().lower()
    if response != 'yes':
        print("[INFO] Aborted. Use 'python scripts/run_data_splitter.py' instead.")
        exit(0)
    
    print("[INFO] RUNNING DATA AUGMENTATION")
    print("Creating augmented dataset in data/augmented/")
    print("=" * 50)
    
    # Updated to 800 per class for better accuracy (80/20 split optimization)
    aug = DataAugmenter(target_per_class=800)
    aug.augment_dataset()
    
    print("\n[SUCCESS] Augmentation completed!")
    print("[INFO] Augmented images saved to: data/augmented/")
    print("[WARN] Remember: Split your data BEFORE using these augmented images!")