from pathlib import Path
import random
import os
import shutil
import argparse

# ✅ Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, help='Path to data folder containing images and labels')
parser.add_argument('--train_pct', type=float, default=0.8, help='Ratio of images for training (example: 0.8 for 80%)')

args = parser.parse_args()

data_path = Path(args.datapath)
train_percent = args.train_pct

# ✅ Check for valid entries
if not data_path.exists():
    raise FileNotFoundError(f"Directory {data_path} not found. Check the path and try again.")
if not (0.01 <= train_percent <= 0.99):
    raise ValueError('Invalid value for --train_pct. Must be between 0.01 and 0.99.')

# ✅ Define input paths
input_image_path = data_path / 'images'
input_label_path = data_path / 'labels'

# ✅ Define YOLO folder structure
output_path = Path('data')
train_img_path = output_path / 'train/images'
train_lbl_path = output_path / 'train/labels'
val_img_path = output_path / 'val/images'
val_lbl_path = output_path / 'val/labels'

# ✅ Create necessary folders
for folder in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
    folder.mkdir(parents=True, exist_ok=True)

# ✅ Get list of all images
image_formats = ['.jpg', '.jpeg', '.png']
img_file_list = list(input_image_path.rglob('*'))
img_file_list = [f for f in img_file_list if f.suffix.lower() in image_formats]

random.shuffle(img_file_list)  # Shuffle to ensure randomness

# ✅ Split dataset into train/validation
split_idx = int(len(img_file_list) * train_percent)
train_files = img_file_list[:split_idx]
val_files = img_file_list[split_idx:]

# ✅ Function to move files
def move_files(file_list, img_dest, lbl_dest):
    for img_path in file_list:
        img_name = img_path.name
        label_name = img_path.stem + '.txt'
        label_path = input_label_path / label_name

        # Move image
        shutil.move(str(img_path), str(img_dest / img_name))

        # Move label if exists
        if label_path.exists():
            shutil.move(str(label_path), str(lbl_dest / label_name))

# ✅ Move train and val files
print(f"Moving {len(train_files)} files to train folder...")
move_files(train_files, train_img_path, train_lbl_path)

print(f"Moving {len(val_files)} files to val folder...")
move_files(val_files, val_img_path, val_lbl_path)

# ✅ Summary
print(f"✅ Train set: {len(train_files)} images")
print(f"✅ Validation set: {len(val_files)} images")
