import os
import shutil
import random

# Settings
dataset_dir = "augmented_dataset"
output_dir = "last_dataset"
image_dir = os.path.join(dataset_dir, "images")
label_dir = os.path.join(dataset_dir, "labels")
split_ratios = (0.8, 0.2)  # (train, val)

# Ensure split ratios sum to 1.0
assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

# Create output directories
splits = ["train", "val"]
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Collect all valid images (only ones with corresponding labels)
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and
    os.path.exists(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))
]

# Shuffle files randomly
random.shuffle(image_files)

# Split files
total_images = len(image_files)
train_count = int(split_ratios[0] * total_images)

train_files = image_files[:train_count]
val_files = image_files[train_count:]

# Copy files into respective folders
for split_name, files in zip(["train", "val"], [train_files, val_files]):
    for image_file in files:
        label_file = os.path.splitext(image_file)[0] + ".txt"

        src_image = os.path.join(image_dir, image_file)
        src_label = os.path.join(label_dir, label_file)

        dst_image = os.path.join(output_dir, "images", split_name, image_file)
        dst_label = os.path.join(output_dir, "labels", split_name, label_file)

        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

print("✅ Dataset split complete!")
print(f"Images saved under {output_dir}/images/[train|val]")
print(f"Labels saved under {output_dir}/labels/[train|val]")