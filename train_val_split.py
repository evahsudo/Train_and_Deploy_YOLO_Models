from pathlib import Path
import shutil
import random
import argparse
import os

# ✅ Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', required=True, help='Path to data folder containing images and labels')
parser.add_argument('--train_pct', type=float, default=0.8, help='Ratio of data for training (0.01 to 0.99)')

args = parser.parse_args()

# ✅ Validate inputs
data_path = Path(args.datapath)
train_pct = args.train_pct

if not data_path.exists():
    raise FileNotFoundError(f"Error: {data_path} does not exist.")
if not (0.01 <= train_pct <= 0.99):
    raise ValueError("Error: --train_pct must be between 0.01 and 0.99")

# ✅ Supported formats
image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
label_formats = {'.txt'}

# ✅ YOLO-compatible folder structure
output_path = Path('data')
train_img_path = output_path / 'train/images'
train_lbl_path = output_path / 'train/labels'
val_img_path = output_path / 'val/images'
val_lbl_path = output_path / 'val/labels'

# ✅ Create necessary folders
for folder in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
    folder.mkdir(parents=True, exist_ok=True)

# ✅ Collect image-label pairs (handle nested folders)
image_files = sorted([f for f in data_path.rglob('*') if f.suffix.lower() in image_formats])
label_files = {f.stem: f for f in data_path.rglob('*') if f.suffix.lower() in label_formats}

# ✅ Match images with labels
dataset = []
for img in image_files:
    label = label_files.get(img.stem)
    dataset.append((img, label))

print(f"Found {len(dataset)} image-label pairs")

# ✅ Split data
random.shuffle(dataset)
split_idx = int(len(dataset) * train_pct)
train_data = dataset[:split_idx]
val_data = dataset[split_idx:]

# ✅ Function to move data
def move_data(data, img_dest, lbl_dest):
    for img, lbl in data:
        # Move image
        dest_img = img_dest / img.name
        if not dest_img.exists():
            shutil.move(str(img), str(dest_img))

        # Move label if it exists
        if lbl and lbl.exists():
            dest_lbl = lbl_dest / lbl.name
            if not dest_lbl.exists():
                shutil.move(str(lbl), str(dest_lbl))

# ✅ Move train and validation data
print(f"Moving {len(train_data)} files to train folder...")
move_data(train_data, train_img_path, train_lbl_path)

print(f"Moving {len(val_data)} files to val folder...")
move_data(val_data, val_img_path, val_lbl_path)

# ✅ Clean up any empty folders
for folder in data_path.rglob('*'):
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()

# ✅ Summary
print(f"✅ Train set: {len(train_data)} images")
print(f"✅ Validation set: {len(val_data)} images")
