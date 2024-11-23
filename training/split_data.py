import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(original_dataset_dir, output_dir, test_size=0.2):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in os.listdir(original_dataset_dir):
        category_dir = os.path.join(original_dataset_dir, category)
        if os.path.isdir(category_dir): 
            train_category_dir = os.path.join(train_dir, category)
            val_category_dir = os.path.join(val_dir, category)

            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(val_category_dir, exist_ok=True)

            images = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

            train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)

            for img in train_images:
                shutil.copy(os.path.join(category_dir, img), os.path.join(train_category_dir, img))
            for img in val_images:
                shutil.copy(os.path.join(category_dir, img), os.path.join(val_category_dir, img))

original_dataset_dir = "C:/Users/micha/Downloads/archive/raw-img"  
output_dir = "C:/Users/micha/OneDrive/Desktop/honours-main/honours-main/training/animals10" 
organize_dataset(original_dataset_dir, output_dir)
