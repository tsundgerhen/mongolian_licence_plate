import os
import cv2
import albumentations as A
import random
from glob import glob
import shutil

# Paths
input_img_dir = "labeled_output_dataset/images"
input_lbl_dir = "labeled_output_dataset/labels"
output_img_dir = "augmented_dataset/images"
output_lbl_dir = "augmented_dataset/labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Augmentations list
augmentations = {
    "snow": A.Compose([A.RandomSnow(p=1.0)]),
    "fog": A.Compose([A.RandomFog(p=1.0)]),
    "bright": A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=0.0, p=1.0)]),
    "dark": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=0.0, p=1.0)]),
    "blur": A.Compose([A.GaussianBlur(blur_limit=3, p=1.0)]),
    "motion_blur": A.Compose([A.MotionBlur(blur_limit=7, p=1.0)]),
    "noise": A.Compose([A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0)]),  # Corrected
    "rain": A.Compose([A.RandomRain(p=1.0)]),
    "contrast": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.4, p=1.0)]),
    "rotate": A.Compose([A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0)]),
    "perspective": A.Compose([A.Perspective(scale=(0.05, 0.1), p=1.0)]),
    "scale": A.Compose([A.RandomScale(scale_limit=0.2, p=1.0)]),
    "crop": A.Compose([
        A.RandomCrop(height=384, width=640, p=1.0)  # fixed here (use RandomCrop)
    ]),
    "clahe": A.Compose([A.CLAHE(p=1.0)]),
    "compression": A.Compose([A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0)]),  # fixed here
    "shadow": A.Compose([A.RandomShadow(p=1.0)]),
    "color_shift": A.Compose([
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
    ]),
}

# Get all image files
image_paths = glob(os.path.join(input_img_dir, "*.jpg")) + glob(os.path.join(input_img_dir, "*.png"))

# Augment images and copy labels
for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(input_lbl_dir, f"{img_name}.txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)

    # Resize image before crop if smaller than the required crop size
    if image.shape[0] < 384 or image.shape[1] < 640:
        image = cv2.resize(image, (640, 384))

    # Copy original image and label
    shutil.copy(img_path, os.path.join(output_img_dir, f"{img_name}.jpg"))
    shutil.copy(label_path, os.path.join(output_lbl_dir, f"{img_name}.txt"))

    # Apply all augmentations
    for aug_name, transform in augmentations.items():
        augmented = transform(image=image)
        aug_img = augmented["image"]

        aug_img_name = f"{img_name}_{aug_name}.jpg"
        aug_lbl_name = f"{img_name}_{aug_name}.txt"

        aug_img_path = os.path.join(output_img_dir, aug_img_name)
        aug_lbl_path = os.path.join(output_lbl_dir, aug_lbl_name)

        # Save augmented image as JPG
        cv2.imwrite(aug_img_path, aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Copy label without changes
        shutil.copy(label_path, aug_lbl_path)

print("✅ All done! Augmented images and labels saved.")