import os
import shutil

#script for translating folder names
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
    "ragno": "spider",
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

dataset_dir = "C:/Users/micha/Downloads/archive/raw-img"

folders = os.listdir(dataset_dir)

for folder in folders:
    if folder in translate:
        old_name = os.path.join(dataset_dir, folder)
        new_name = os.path.join(dataset_dir, translate[folder])

        # Rename the folder
        os.rename(old_name, new_name)
        print(f"Renamed folder: {old_name} -> {new_name}")
    else:
        print(f"No translation for folder: {folder}")
