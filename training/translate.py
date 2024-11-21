import os
import shutil

# Import the translation dictionary from your translate.py file
translate = {
    "cane": "dog", 
    "cavallo": "horse", 
    "elefante": "elephant", 
    "farfalla": "butterfly", 
    "gallina": "chicken", 
    "gatto": "cat", 
    "mucca": "cow", 
    "pecora": "sheep", 
    "scoiattolo": "squirrel", 
    "ragno": "spider",  # Add this line for "ragno"
    "dog": "cane", 
    "horse": "cavallo", 
    "elephant": "elefante", 
    "butterfly": "farfalla", 
    "chicken": "gallina", 
    "cat": "gatto", 
    "cow": "mucca", 
    "spider": "ragno", 
    "squirrel": "scoiattolo"
}

# Define the root directory where your dataset is located
dataset_dir = "C:/Users/micha/Downloads/archive/raw-img"

# Get the list of folders in the dataset directory
folders = os.listdir(dataset_dir)

# Loop through each folder in the directory
for folder in folders:
    # Check if the folder name is in the translation dictionary
    if folder in translate:
        # Construct the old and new folder names
        old_name = os.path.join(dataset_dir, folder)
        new_name = os.path.join(dataset_dir, translate[folder])

        # Rename the folder
        os.rename(old_name, new_name)
        print(f"Renamed folder: {old_name} -> {new_name}")
    else:
        print(f"No translation for folder: {folder}")