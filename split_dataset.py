import os
import random
import shutil
from pathlib import Path

def split_dataset(images_dir, labels_dir, train_dir, val_dir, train_ratio=0.8):
    """
    Split dataset into train and validation sets.

    Args:
        images_dir (str): Path to the directory containing images.
        labels_dir (str): Path to the directory containing labels.
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        train_ratio (float): Ratio of training data (default: 0.8).
    """
    # Get list of all image files
    image_files = list(Path(images_dir).glob("*"))
    
    # Shuffle the list
    random.shuffle(image_files)
    
    # Calculate split index
    split_index = int(len(image_files) * train_ratio)
    
    # Split into train and validation sets
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]
    
    # Create symlinks for training set
    for img_path in train_images:
        # Create symlink for image
        img_dst = Path(train_dir) / "images" / img_path.name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        img_dst.symlink_to(img_path)
        
        # Create symlink for corresponding label
        label_name = img_path.stem + ".txt"
        label_path = Path(labels_dir) / label_name
        if label_path.exists():
            label_dst = Path(train_dir) / "labels" / label_name
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.symlink_to(label_path)
    
    # Create symlinks for validation set
    for img_path in val_images:
        # Create symlink for image
        img_dst = Path(val_dir) / "images" / img_path.name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        img_dst.symlink_to(img_path)
        
        # Create symlink for corresponding label
        label_name = img_path.stem + ".txt"
        label_path = Path(labels_dir) / label_name
        if label_path.exists():
            label_dst = Path(val_dir) / "labels" / label_name
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.symlink_to(label_path)

if __name__ == "__main__":
    # Define paths
    merged_dataset_dir = Path("/root/autodl-tmp/three_data/merged_dataset")
    images_dir = merged_dataset_dir / "images"
    labels_dir = merged_dataset_dir / "labels"
    train_dir = merged_dataset_dir / "train"
    val_dir = merged_dataset_dir / "val"
    
    # Create train and validation directories if they don't exist
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_dir / "labels").mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)
    (val_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    split_dataset(images_dir, labels_dir, train_dir, val_dir)
    
    print("Dataset split completed successfully!")