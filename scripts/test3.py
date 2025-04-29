from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ==== Utils ====

def draw_mongolian_text_on_image(image, text, position, font_path="../Roboto/static/Roboto-Regular.ttf", font_size=32, color=(0, 255, 0)):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", font_size)
    draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def label_to_char(cls_id):
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М',
                'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ',
                'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ө', 'Ү']
    return alphabet[int(cls_id)] if 0 <= int(cls_id) < len(alphabet) else '?'

def fix_plate_format(plate_text):
    digits = ''.join([c for c in plate_text if c.isdigit()])
    letters = ''.join([c for c in plate_text if c.isalpha()])
    return digits[:4] + letters[:3]  # Expected format: 4 digits + 3 letters

# ==== Load Models ====
plate_model = YOLO("../models/plate_model.pt")
char_model = YOLO("../models/best4.pt")

# ==== Load Image ====
image_path = "../scraped_plates/mn_0008-ХОА_1.jpg"
image = cv2.imread(image_path)

# ==== Detect Plates ====
plate_results = plate_model(image)[0]

for plate in plate_results.boxes:
    x, y, w, h = plate.xywh[0].tolist()
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    plate_crop = image[y1:y2, x1:x2]

    # ==== Detect Characters in Plate Crop ====
    char_results = char_model(plate_crop)[0]
    characters = []

    for box, cls, conf in zip(char_results.boxes.xyxy, char_results.boxes.cls, char_results.boxes.conf):
        if conf < 0.1:
            continue

        cx1, cy1, cx2, cy2 = map(int, box)
        char_label = label_to_char(cls)
        cx = (cx1 + cx2) // 2
        characters.append((cx, char_label, conf))  # Use center x and confidence

        # Print the character and its confidence
        print(f"Detected Character: {char_label}, Confidence: {conf:.2f}")

        # Annotate on the plate_crop
        plate_crop_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(plate_crop_pil)
        font = ImageFont.truetype("../Roboto/static/Roboto-Regular.ttf", 18)
        draw.rectangle([cx1, cy1, cx2, cy2], outline=(255, 0, 0), width=1)
        draw.text((cx1, cy1 - 20), f"{char_label} {conf:.2f}", font=font, fill=(255, 0, 0))
        plate_crop = cv2.cvtColor(np.array(plate_crop_pil), cv2.COLOR_RGB2BGR)

    # ==== Sort & Format Using Top 7 Most Confident Characters ====
    characters.sort(key=lambda x: x[2], reverse=True)  # sort by confidence
    top_characters = characters[:7]
    top_characters.sort(key=lambda x: x[0])  # sort by x (horizontal position)
    raw_plate_text = ''.join([c for _, c, _ in top_characters])
    plate_text = fix_plate_format(raw_plate_text)

    if len(plate_text) >= 6:
        print(f"Detected Plate: {plate_text}")
        image = draw_mongolian_text_on_image(image, plate_text, (x1, y1 - 30))

    # Draw plate box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    image[y1:y2, x1:x2] = plate_crop

# ==== Show & Save Result ====
cv2.imshow("Detected Plate with Characters", image)
cv2.waitKey(0)
cv2.destroyAllWindows()