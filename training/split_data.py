import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(original_dataset_dir, output_dir, test_size=0.2):
    # Create base train/val directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in os.listdir(original_dataset_dir):
        category_dir = os.path.join(original_dataset_dir, category)
        if os.path.isdir(category_dir):  # Check if it's a folder (category)
            train_category_dir = os.path.join(train_dir, category)
            val_category_dir = os.path.join(val_dir, category)

            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(val_category_dir, exist_ok=True)

            # Get all images in the category folder
            images = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

            # Split into train and validation sets
            train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)

            # Copy images to the appropriate directories
            for img in train_images:
                shutil.copy(os.path.join(category_dir, img), os.path.join(train_category_dir, img))
            for img in val_images:
                shutil.copy(os.path.join(category_dir, img), os.path.join(val_category_dir, img))

# Example usage:
# Replace the paths with your dataset location
original_dataset_dir = "C:/Users/micha/Downloads/archive/raw-img"  # Replace with actual path
output_dir = "C:/Users/micha/OneDrive/Desktop/honours-main/honours-main/training/animals10"  # Replace with desired output path
organize_dataset(original_dataset_dir, output_dir)
