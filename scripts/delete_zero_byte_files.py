import os
import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import RAW_DATA_DIR, CLASS_ID_MAP

def delete_zero_byte_files(data_dir=RAW_DATA_DIR):
    """
    Delete zero-byte (corrupted) files from all class folders.
    
    Args:
        data_dir: Root directory containing class folders (default: data/raw)
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found: {data_dir}")
        print(f"   Please create the directory and add your class folders:")
        print(f"   {data_dir}/")
        for class_name in CLASS_ID_MAP.keys():
            print(f"     └── {class_name}/")
        return
    
    total_deleted = 0
    
    print(f"[INFO] Checking for zero-byte files in {data_dir}/")
    print("=" * 50)
    
    for class_name in CLASS_ID_MAP.keys():
        class_folder = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_folder):
            print(f"[WARN] Skipping {class_name}: folder not found")
            continue
        
        if not os.path.isdir(class_folder):
            print(f"[WARN] Skipping {class_name}: not a directory")
            continue
        
        deleted_count = 0
        files_checked = 0
        
        try:
            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                files_checked += 1
                
                # Check if file is zero bytes
                if os.path.getsize(file_path) == 0:
                    try:
                        os.remove(file_path)
                        print(f"  [DELETED] {class_name}/{filename}")
                        deleted_count += 1
                        total_deleted += 1
                    except Exception as e:
                        print(f"  [ERROR] Error deleting {file_path}: {e}")
        
        except Exception as e:
            print(f"  [ERROR] Error reading {class_folder}: {e}")
            continue
        
        if deleted_count > 0:
            print(f"  [OK] {class_name}: Deleted {deleted_count} zero-byte file(s) out of {files_checked} checked")
        else:
            print(f"  [OK] {class_name}: No zero-byte files found ({files_checked} files checked)")
    
    print("=" * 50)
    if total_deleted > 0:
        print(f"[SUCCESS] Cleanup complete! Deleted {total_deleted} zero-byte file(s) total.")
    else:
        print("[SUCCESS] No zero-byte files found. All files are valid!")


if __name__ == "__main__":
    delete_zero_byte_files()
