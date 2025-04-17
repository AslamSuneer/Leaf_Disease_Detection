import shutil
from pathlib import Path

# Define the data spilt -> train/vaild
split_name = 'valid'
# Define paths
data_path = Path("./data")
images_train_path = data_path / "images" / split_name
labels_train_path = data_path / "labels" / split_name

# Ensure the train folders exist
images_train_path.mkdir(parents=True, exist_ok=True)
labels_train_path.mkdir(parents=True, exist_ok=True)

# Move all images from subfolders to "train" folder
for class_folder in images_train_path.glob("*/"):  # Loop through subfolders
    if class_folder.is_dir():
        for image in class_folder.glob("*.*"):  # Move all image files
            shutil.move(str(image), str(images_train_path / image.name))
        class_folder.rmdir()  # Remove empty folder

# Move all labels from subfolders to "train" folder
for class_folder in labels_train_path.glob("*/"):
    if class_folder.is_dir():
        for label in class_folder.glob("*.txt"):  # Move label files
            shutil.move(str(label), str(labels_train_path / label.name))
        class_folder.rmdir()  # Remove empty folder

print("Dataset structure flattened successfully! 🚀")
