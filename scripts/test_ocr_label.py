import cv2
import os
import pandas as pd
import pytesseract
from pytesseract import Output
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Parameters
preview_count = 5

# Paths
image_folder = "mlub-mongolian-car-plate-prediction/test/test"
csv_file_path = "mlub-mongolian-car-plate-prediction/test.csv"
font_path = "../Roboto/static/Roboto-Bold.ttf"  # 📝 Or replace with Mongolian font

# Read CSV
df = pd.read_csv(csv_file_path)

# Define Mongolian characters
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И',
         'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т',
         'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь',
         'Э', 'Ю', 'Я', 'Ө', 'Ү']
char_to_id = {char: idx for idx, char in enumerate(chars)}

# Preview labeling
count = 0
for index, row in df.iterrows():
    if count >= preview_count:
        break

    filename = row['file_name']
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {filename}")
        continue

    h, w = image.shape[:2]

    try:
        data = pytesseract.image_to_boxes(image, lang='mon', output_type=Output.DICT)
    except Exception as e:
        print(f"Tesseract failed on {filename}: {e}")
        continue

    if 'char' not in data or len(data['char']) == 0:
        print(f"No characters detected in {filename}")
        continue

    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, 28)

    for i in range(len(data['char'])):
        char = data['char'][i].upper()
        if char not in char_to_id:
            continue
        x1, y1, x2, y2 = int(data['left'][i]), int(data['bottom'][i]), int(data['right'][i]), int(data['top'][i])
        y1, y2 = h - y1, h - y2  # Tesseract to OpenCV coordinate fix

        # Draw box and Unicode label using PIL
        draw.rectangle([x1, y2, x2, y1], outline="green", width=2)
        draw.text((x1, y2 - 30), f"{char}", font=font, fill=(255, 255, 0))

    # Convert back to OpenCV format
    image_labeled = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    print(f"Previewing labeled characters for: {filename}")
    cv2.imshow("Labeled Image", image_labeled)
    key = cv2.waitKey(0)
    if key == 27:  # Press Esc to break early
        break

    count += 1

cv2.destroyAllWindows()