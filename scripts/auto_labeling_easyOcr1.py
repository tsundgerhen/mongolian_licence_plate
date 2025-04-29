import os
import cv2
import re
import pytesseract
from pytesseract import Output
from ultralytics import YOLO

# Set paths
image_folder = "../scraped_plates"  # full car images
output_dataset_dir = "cropped_plate_dataset"
output_images_dir = os.path.join(output_dataset_dir, "images")
output_labels_dir = os.path.join(output_dataset_dir, "labels")
plate_model_path = "../models/plate_model.pt"  # your YOLOv8 plate detection model

# Make output dirs
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Character map
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И',
         'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т',
         'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь',
         'Э', 'Ю', 'Я', 'Ө', 'Ү']
char_to_id = {char: idx for idx, char in enumerate(chars)}

# Load plate detector
plate_model = YOLO(plate_model_path)

# Helper to extract expected plate number
def extract_plate_from_filename(filename):
    name = os.path.splitext(filename)[0]
    match = re.search(r'\d{4}-?([А-ЯӨҮЁ]{3})', name)
    return match.group(0).replace("-", "") if match else None

# Helper to clean OCR results
def clean_ocr_text(text):
    return re.sub(r'\s+', '', text.upper())

# Process all images
for filename in os.listdir(image_folder):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    expected_plate = extract_plate_from_filename(filename.upper())
    if not expected_plate:
        print(f"Skipping {filename}, no valid plate in filename")
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {filename}")
        continue

    results = plate_model(image)
    plates = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(plates) == 0:
        print(f"No plate found in {filename}")
        continue

    # Use first plate found
    x1, y1, x2, y2 = map(int, plates[0])
    plate_crop = image[y1:y2, x1:x2]
    if plate_crop.size == 0:
        print(f"Invalid crop for {filename}")
        continue

    h, w = plate_crop.shape[:2]

    # OCR on plate crop
    try:
        ocr_result = pytesseract.image_to_string(plate_crop, lang='mon')
        ocr_text = clean_ocr_text(ocr_result)
    except Exception as e:
        print(f"Tesseract failed on {filename}: {e}")
        continue

    if not ocr_text:
        print(f"OCR failed to detect text in {filename}")
        continue

    # Check if OCR matched expected plate exactly
    if ocr_text != expected_plate:
        print(f"OCR mismatch in {filename}: Expected {expected_plate}, Got {ocr_text}")
        continue

    # Now OCR by boxes for labeling
    try:
        data = pytesseract.image_to_boxes(plate_crop, lang='mon', output_type=Output.DICT)
    except Exception as e:
        print(f"Tesseract boxes failed on {filename}: {e}")
        continue

    if 'char' not in data or len(data['char']) == 0:
        print(f"No character boxes found for {filename}")
        continue

    # Save crop
    save_img_name = os.path.splitext(filename)[0] + ".jpg"
    save_img_path = os.path.join(output_images_dir, save_img_name)
    cv2.imwrite(save_img_path, plate_crop)

    # Save labels
    save_lbl_path = os.path.join(output_labels_dir, os.path.splitext(filename)[0] + ".txt")
    with open(save_lbl_path, 'w') as f:
        for i in range(len(data['char'])):
            char = data['char'][i].upper()
            if char not in char_to_id:
                continue

            x1 = int(data['left'][i])
            y1 = h - int(data['top'][i])
            x2 = int(data['right'][i])
            y2 = h - int(data['bottom'][i])

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"{char_to_id[char]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Saved correctly labeled: {filename} -> {expected_plate}")

    # --- OPTIONAL: Show preview ---
    preview = plate_crop.copy()
    for i in range(len(data['char'])):
        char = data['char'][i].upper()
        if char not in char_to_id:
            continue

        x1 = int(data['left'][i])
        y1 = h - int(data['top'][i])
        x2 = int(data['right'][i])
        y2 = h - int(data['bottom'][i])

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, char, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Labeled Plate Preview", preview)
    key = cv2.waitKey(0)
    if key == 27:  # Press Esc to exit preview early
        break
    cv2.destroyAllWindows()

# Save class info
data_yaml = f"""
train: ../{output_dataset_dir}/images
val: ../{output_dataset_dir}/images
nc: {len(chars)}
names: {chars}
"""
with open(os.path.join(output_dataset_dir, "data.yaml"), 'w') as f:
    f.write(data_yaml)

print("Finished labeling full car images.")