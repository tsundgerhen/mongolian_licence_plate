import cv2
import os
import pandas as pd
import pytesseract
from pytesseract import Output

# --------- Configuration ---------
image_folder = "mlub-mongolian-car-plate-prediction/test/test"
csv_file_path = "mlub-mongolian-car-plate-prediction/test.csv"
output_dataset_dir = "labeled_output_dataset"
output_images_dir = os.path.join(output_dataset_dir, "images")
output_labels_dir = os.path.join(output_dataset_dir, "labels")

# --------- Create folders if not exist ---------
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# --------- Mongolian Character Classes ---------
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И',
         'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т',
         'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь',
         'Э', 'Ю', 'Я', 'Ө', 'Ү']
char_to_id = {char: idx for idx, char in enumerate(chars)}

# --------- Load CSV ---------
df = pd.read_csv(csv_file_path)

# --------- Process Each Image ---------
for index, row in df.iterrows():
    filename = row['file_name']
    expected_plate = row['plate_number'].replace("-", "").upper()

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {filename}")
        continue

    h, w = image.shape[:2]

    try:
        data = pytesseract.image_to_boxes(image, lang='mon', output_type=Output.DICT)
    except Exception as e:
        print(f"❌ Tesseract failed on {filename}: {e}")
        continue

    if 'char' not in data or len(data['char']) == 0:
        print(f"⚠️ No characters detected in {filename}")
        continue

    # --------- Collect detected characters ---------
    detected_chars = []
    boxes = []

    for i in range(len(data['char'])):
        char = data['char'][i].upper()
        if char in char_to_id:
            detected_chars.append(char)
            boxes.append((int(data['left'][i]), int(data['bottom'][i]),
                          int(data['right'][i]), int(data['top'][i])))

    detected_string = ''.join(detected_chars)

    # --------- Strict match: detected number part must match expected number part ---------
    expected_numbers = ''.join([c for c in expected_plate if c.isdigit()])
    detected_numbers = ''.join([c for c in detected_chars if c.isdigit()])

    if expected_numbers != detected_numbers:
        print(f"⚠️ Skipped {filename}: number mismatch (expected {expected_numbers}, detected {detected_numbers})")
        continue  # Skip this image if numbers don't match perfectly

    # --------- If matched, save label ---------
    lines = []
    for char, (x1, y1, x2, y2) in zip(detected_chars, boxes):
        y1, y2 = h - y1, h - y2  # Convert to top-left origin
        center_x = (x1 + x2) / 2 / w
        center_y = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y1 - y2) / h
        class_id = char_to_id[char]
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(output_labels_dir, label_filename)

    # Save labels and images
    if not os.path.exists(label_path):
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        output_image_path = os.path.join(output_images_dir, filename)
        cv2.imwrite(output_image_path, image)

    print(f"✅ Labeled {filename}: {len(lines)} characters (numbers matched)")

# --------- Create data.yaml ---------
data_yaml = f"""
train: ../{output_dataset_dir}/images
val: ../{output_dataset_dir}/images
nc: {len(chars)}
names: {chars}
"""

with open(os.path.join(output_dataset_dir, "data.yaml"), 'w', encoding='utf-8') as f:
    f.write(data_yaml)

print("\n🎉 Auto-labeling complete. Output saved to:", output_dataset_dir)