import os
import shutil
import numpy as np

def split_normalised_pointclouds(normalized_dir, output_base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits normalized point cloud .txt files into train, val, and test directories.
    
    Args:
        normalized_dir: Path to directory containing all normalized .txt files
        output_base_dir: Base path where train/, val/, test/ subdirectories will be created
        train_ratio: Fraction for training set (default 0.8)
        val_ratio: Fraction for validation set (default 0.1)
        test_ratio: Fraction for test set (default 0.1)
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    test_dir = os.path.join(output_base_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all .txt files
    txt_files = [f for f in os.listdir(normalized_dir) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} .txt files")
    
    # Shuffle files randomly
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(txt_files)
    
    # Calculate split indices
    n_files = len(txt_files)
    train_count = int(np.ceil(n_files * train_ratio))
    val_count = int(np.ceil(n_files * val_ratio))
    
    train_files = txt_files[:train_count]
    val_files = txt_files[train_count:train_count + val_count]
    test_files = txt_files[train_count + val_count:]
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")
    
    # Copy files to respective directories
    for filename in train_files:
        src = os.path.join(normalized_dir, filename)
        dst = os.path.join(train_dir, filename)
        shutil.copy2(src, dst)
    
    for filename in val_files:
        src = os.path.join(normalized_dir, filename)
        dst = os.path.join(val_dir, filename)
        shutil.copy2(src, dst)
    
    for filename in test_files:
        src = os.path.join(normalized_dir, filename)
        dst = os.path.join(test_dir, filename)
        shutil.copy2(src, dst)
    
    print("Split completed successfully!")

# === Configuration ===
# Path to directory containing all normalized .txt files
NORMALIZED_FOLDER = "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/Normalised/T2"
OUTPUT_BASE_FOLDER = "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/Normalised"


if __name__ == "__main__":
    split_normalised_pointclouds(NORMALIZED_FOLDER, OUTPUT_BASE_FOLDER, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
