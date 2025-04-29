import os
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import re

# Load character detection model
char_model = YOLO('../models/best4.pt')

# Mapping from class indices to characters
char_map = {i: str(i) for i in range(10)}
char_map.update({
    10: 'А', 11: 'Б', 12: 'В', 13: 'Г', 14: 'Д', 15: 'Е', 16: 'Ё', 17: 'Ж', 18: 'З', 19: 'И',
    20: 'Й', 21: 'К', 22: 'Л', 23: 'М', 24: 'Н', 25: 'О', 26: 'П', 27: 'Р', 28: 'С', 29: 'Т',
    30: 'У', 31: 'Ф', 32: 'Х', 33: 'Ц', 34: 'Ч', 35: 'Ш', 36: 'Щ', 37: 'Ъ', 38: 'Ы', 39: 'Ь',
    40: 'Э', 41: 'Ю', 42: 'Я', 43: 'Ө', 44: 'Ү'
})

# Paths
image_folder = 'mlub-mongolian-car-plate-prediction/training/training'
csv_path = 'mlub-mongolian-car-plate-prediction/training.csv'

# Load CSV
df = pd.read_csv(csv_path)

correct = 0
total = 0

def apply_clahe(image):
    """Enhance contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def denoise_and_sharpen(image):
    """Denoise and apply sharpening filter."""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def clean_and_validate_plate(predicted):
    """
    Cleans and validates the predicted plate number.
    Tries to fix common mistakes and ensure pattern validity.
    """
    fixes = {
        'О': '0', '0': '0',
        'В': '8', '8': '8',
        'С': 'C', 'З': '3',
        'Т': 'T', 'А': 'A',
    }
    cleaned = ''.join(fixes.get(ch, ch) for ch in predicted)

    # Valid Mongolian pattern: 4 digits + 3 Mongolian capital letters
    if re.match(r'^\d{4}[А-ЯӨҮ]{3}$', cleaned):
        return cleaned
    else:
        return predicted  # fallback to original if invalid

for index, row in df.iterrows():
    file_name = row['file_name']
    true_plate = row['plate_number']

    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {file_name}")
        continue

    # Resize image
    image_resized = cv2.resize(image, (416, 256))

    # Preprocessing
    image_enhanced = apply_clahe(image_resized)
    image_enhanced = denoise_and_sharpen(image_enhanced)

    # Upscale
    image_upscaled = cv2.resize(image_enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Predict characters
    results = char_model(image_upscaled, conf=0.32)
    chars = []
    for r in results:
        for box in r.boxes:
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            cls = int(box.cls[0])
            chars.append((x_center, char_map.get(cls, '?')))

    # Sort and predict
    chars.sort(key=lambda x: x[0])
    raw_predicted_plate = ''.join([c[1] for c in chars])
    predicted_plate = clean_and_validate_plate(raw_predicted_plate)

    is_correct = predicted_plate == true_plate
    total += 1
    correct += int(is_correct)

    print(f"{file_name}: Predicted: {predicted_plate}, Actual: {true_plate}, {'✅' if is_correct else '❌'}")

# Accuracy summary
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

# Note: Further improve by fine-tuning char_model on real video samples.