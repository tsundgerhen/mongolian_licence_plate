import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ==== Utils ====
def label_to_char(cls_id):
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ө', 'Ү']
    return alphabet[int(cls_id)] if 0 <= int(cls_id) < len(alphabet) else '?'

def sort_left_to_right(boxes):
    return sorted(boxes, key=lambda b: b[0])  # sort by x center

def draw_mongolian_text_on_image(image, text, position, font_path="Roboto/static/Roboto-Regular.ttf", font_size=32, color=(0, 255, 0)):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def fix_plate_format(plate_text):
    digits = ''.join([c for c in plate_text if c.isdigit()])
    letters = ''.join([c for c in plate_text if c.isalpha()])
    return digits[:4] + letters[:3]  # Expected format: 4 digits + 3 letters

# ==== Load Models ====
plate_model = YOLO("../models/plate_model.pt")
char_model = YOLO("../models/best4.pt")

# ==== Folder containing test images ====
test_image_folder = "../plate_detected_images"  # Folder with test images
correct_count = 0
total_count = 0

# Allowed characters based on training
allowed_characters = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ө', 'Ү'])

# ==== Process Each Image in the Folder ====
for filename in os.listdir(test_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        total_count += 1
        image_path = os.path.join(test_image_folder, filename)
        image = cv2.imread(image_path)

        # ==== Extract Plate Number from Filename ====
        plate_number_parts = filename.split('_')[1:-1]  # Split and skip the first and last parts
        actual_plate_number = ''.join(plate_number_parts)  # Concatenate the parts to form the full plate number
        actual_plate_number = actual_plate_number.replace("-", "")  # Remove hyphen for comparison
        actual_plate_number = fix_plate_format(actual_plate_number)  # <<=== FIX FORMAT for ground truth

        # ==== Detect Plates ====
        plate_results = plate_model(image)[0]
        detected_plate_text = ""

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
                if conf < 0.3:  # Confidence threshold
                    continue

                cx1, cy1, cx2, cy2 = map(int, box)
                cx_center = (cx1 + cx2) // 2
                char_label = label_to_char(cls)

                if char_label in allowed_characters:
                    characters.append((cx_center, char_label))

                # Draw character box and confidence
                cv2.rectangle(plate_crop, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
                cv2.putText(plate_crop, f"{char_label} {conf:.2f}", (cx1, cy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # ==== Sort & Build Plate Text ====
            sorted_chars = sort_left_to_right(characters)
            detected_plate_text = ''.join([c for _, c in sorted_chars])
            detected_plate_text = fix_plate_format(detected_plate_text)  # <<=== FIX FORMAT for detected

            # Draw plate box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Replace the cropped region with annotated version
            image[y1:y2, x1:x2] = plate_crop

        # ==== Evaluate Accuracy ====
        print(f"Expected Plate Number: {actual_plate_number}")
        print(f"Detected Plate Number: {detected_plate_text}")

        if detected_plate_text == actual_plate_number:
            correct_count += 1

# ==== Calculate and display accuracy ====
accuracy = (correct_count / total_count) * 100
print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")

# ==== Save and Show Annotated Result for Last Image ====
cv2.imwrite("annotated_result.jpg", image)
cv2.imshow("Detected Plate with Characters", image)
cv2.waitKey(0)
cv2.destroyAllWindows()