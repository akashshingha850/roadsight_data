import os
import shutil
from sklearn.model_selection import train_test_split

# Define dataset directory
data_dir = "/home/ubuntu/thesis-Intersection/yolo/data"

# Assume images are in the parent data folder; adjust if different
all_images_dir = data_dir

# Get all image files (assuming .jpg; update if needed)
image_files = [f for f in os.listdir(os.path.join(all_images_dir)) if f.endswith(".jpg")]

# Shuffle and split using sklearn for precise ratios (70/15/15) with fixed seed
train_files, temp = train_test_split(image_files, test_size=0.3, random_state=42) # 70% train 
val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)  # 15% val, 15% test

# Create train, val, and test directories if they don't exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)

# Function to move files
def move_files(file_list, split):
    for file in file_list:
        img_src = os.path.join(all_images_dir, file)
        lbl_src = os.path.join(all_images_dir, file.replace(".jpg", ".txt"))

        img_dst = os.path.join(data_dir, split, "images", file)
        lbl_dst = os.path.join(data_dir, split, "labels", file.replace(".jpg", ".txt"))

        # Move image
        shutil.move(img_src, img_dst)

        # Move label if exists
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)

# Move files to respective directories
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split and reorganization complete!")
