from pathlib import Path
import random
import shutil
import argparse

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files', required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder (example: ".8")', type=float, default=0.8)
parser.add_argument('--output_path', help='Path to output dataset folder', default='./data')

args = parser.parse_args()

data_path = Path(args.datapath)
train_percent = args.train_pct
output_path = Path(args.output_path)

# Check for valid entries
if not data_path.is_dir():
    raise FileNotFoundError(f'Directory "{data_path}" not found. Verify the path is correct.')
if not (0.01 <= train_percent <= 0.99):
    raise ValueError('Invalid entry for train_pct. Please enter a number between 0.01 and 0.99.')

# Define paths to input dataset
input_image_path = data_path / 'images'
input_label_path = data_path / 'labels'

# Define paths to output folders
train_img_path = output_path / 'train/images'
train_txt_path = output_path / 'train/labels'
val_img_path = output_path / 'validation/images'
val_txt_path = output_path / 'validation/labels'

# Create output directories
for path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
    path.mkdir(parents=True, exist_ok=True)

# Get list of all images and annotation files
img_file_list = sorted(list(input_image_path.glob('*')))
txt_file_list = sorted(list(input_label_path.glob('*')))
print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# Shuffle data for randomness
data_pairs = []
for img_path in img_file_list:
    base_fn = img_path.stem
    txt_path = input_label_path / f"{base_fn}.txt"
    data_pairs.append((img_path, txt_path if txt_path.exists() else None))

random.shuffle(data_pairs)

# Split data
train_size = int(len(data_pairs) * train_percent)
train_data = data_pairs[:train_size]
val_data = data_pairs[train_size:]

print(f'Images moving to train: {len(train_data)}')
print(f'Images moving to validation: {len(val_data)}')

# Function to move files
def move_files(data, img_dest, txt_dest):
    for img_path, txt_path in data:
        shutil.copy(img_path, img_dest / img_path.name)
        if txt_path:
            shutil.copy(txt_path, txt_dest / txt_path.name)

# Move files to train and validation sets
move_files(train_data, train_img_path, train_txt_path)
move_files(val_data, val_img_path, val_txt_path)

print("✅ Dataset successfully split into train and validation sets!")
