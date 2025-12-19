"""
Setup script to create the required data directory structure.
"""
import os
import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import RAW_DATA_DIR, CLASS_ID_MAP

def setup_data_structure():
    """Create the data directory structure."""
    print("[INFO] Setting up data directory structure...")
    print("=" * 50)
    
    # Create main data directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"[OK] Created: {RAW_DATA_DIR}/")
    
    # Create class directories
    for class_name in CLASS_ID_MAP.keys():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"[OK] Created: {class_dir}/")
    
    print("=" * 50)
    print("[SUCCESS] Data structure setup complete!")
    print(f"\n[INFO] Next steps:")
    print(f"   1. Add your images to the folders in {RAW_DATA_DIR}/")
    print(f"   2. Run: python scripts/delete_zero_byte_files.py")
    print(f"   3. Run: python scripts/run_data_splitter.py")

if __name__ == "__main__":
    setup_data_structure()

