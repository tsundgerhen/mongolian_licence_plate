import os
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import Counter

# Load character detection model
char_model = YOLO('../models/best4.pt')

# Character mapping
char_map = {i: str(i) for i in range(10)}
char_map.update({
    10: 'А', 11: 'Б', 12: 'В', 13: 'Г', 14: 'Д', 15: 'Е', 16: 'Ё', 17: 'Ж', 18: 'З', 19: 'И',
    20: 'Й', 21: 'К', 22: 'Л', 23: 'М', 24: 'Н', 25: 'О', 26: 'П', 27: 'Р', 28: 'С', 29: 'Т',
    30: 'У', 31: 'Ф', 32: 'Х', 33: 'Ц', 34: 'Ч', 35: 'Ш', 36: 'Щ', 37: 'Ъ', 38: 'Ы', 39: 'Ь',
    40: 'Э', 41: 'Ю', 42: 'Я', 43: 'Ө', 44: 'Ү'
})

# Similar/confused characters map: (low_conf_char, replacement_char)
confusion_map = {
    'А': 'Д',  # If low confidence, assume it could be 'Д'
    'О': 'Ө',
    'Р': 'В',
    'Е': 'Ё',
    'Ү': 'Ц',
    'Н': 'И'
}

# Confidence threshold below which to try correction
CONF_THRESHOLD = 0.325

# Paths
image_folder = 'plates'
csv_path = 'plates.csv'

# Load CSV
df = pd.read_csv(csv_path)

correct = 0
total = 0

for index, row in df.iterrows():
    file_name = row['file_name']
    true_plate = row['plate_number']
    
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {file_name}")
        continue

    image_resized = cv2.resize(image, (416, 256))
    results = char_model(image_resized, conf=0.25)
    chars = []
    for r in results:
        for box in r.boxes:
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            char = char_map.get(cls, '?')

            # Apply low confidence correction
            if conf < CONF_THRESHOLD and char in confusion_map:
                char = confusion_map[char]

            chars.append((x_center, char))

    # Sort by horizontal position
    chars.sort(key=lambda x: x[0])
    predicted_plate = ''.join([c[1] for c in chars])

    is_correct = predicted_plate == true_plate
    total += 1
    correct += int(is_correct)

    print(f"{file_name}: Predicted: {predicted_plate}, Actual: {true_plate}, {'✅' if is_correct else '❌'}")

accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")