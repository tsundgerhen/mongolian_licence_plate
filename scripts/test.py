import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

def draw_mongolian_text_on_image(image, text, position, font_path="../Roboto/static/Roboto-Regular.ttf", font_size=32, color=(0, 255, 0)):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file {font_path} not found. Using default font.")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def fix_plate_format(plate_text):
    digits = ''.join([c for c in plate_text if c.isdigit()])
    letters = ''.join([c for c in plate_text if c.isalpha()])
    return (digits[:4] + letters[:3])[:7]

def get_most_common_plate(track_id, history):
    full_texts = [fix_plate_format(t) for t in history[track_id] if len(fix_plate_format(t)) == 7]
    if not full_texts:
        return ""
    most_common = Counter(full_texts).most_common(1)[0][0]
    return most_common

# Load models
plate_model = YOLO('../models/plate_model.pt')
char_model = YOLO('../models/best4.pt')

# Initialize video capture and writer
cap = cv2.VideoCapture('../test_video3.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('output_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (height, width))

# History for character detections
char_history = defaultdict(list)

# Character mapping
char_map = {i: str(i) for i in range(10)}
char_map.update({
    10: 'А', 11: 'Б', 12: 'В', 13: 'Г', 14: 'Д', 15: 'Е', 16: 'Ё', 17: 'Ж', 18: 'З', 19: 'И',
    20: 'Й', 21: 'К', 22: 'Л', 23: 'М', 24: 'Н', 25: 'О', 26: 'П', 27: 'Р', 28: 'С', 29: 'Т',
    30: 'У', 31: 'Ф', 32: 'Х', 33: 'Ц', 34: 'Ч', 35: 'Ш', 36: 'Щ', 37: 'Ъ', 38: 'Ы', 39: 'Ь',
    40: 'Э', 41: 'Ю', 42: 'Я', 43: 'Ө', 44: 'Ү'
})

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track plates
    plate_results = plate_model.track(frame, conf=0.5, persist=True)
    for r in plate_results:
        for box in r.boxes:
            if box.id is not None:
                track_id = box.id.item()  # Get track ID as integer
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue

                # Detect characters with higher confidence
                char_results = char_model(plate_img, conf=0.4)
                chars = []
                for char_box in char_results[0].boxes:
                    cx = int((char_box.xyxy[0][0] + char_box.xyxy[0][2]) / 2)
                    cls = int(char_box.cls[0])
                    chars.append((cx, char_map.get(cls, '?')))
                chars.sort(key=lambda x: x[0])
                plate_text = ''.join(c[1] for c in chars)
                if plate_text:
                    char_history[track_id].append(plate_text)

                # Get the most common plate text
                final_plate = get_most_common_plate(track_id, char_history)
                if final_plate:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    frame = draw_mongolian_text_on_image(
                        image=frame,
                        text=f'Plate: {final_plate}',
                        position=(x1, max(0, y1 - 50)),
                        font_path="../Roboto/static/Roboto-Regular.ttf",
                        font_size=36,
                        color=(0, 255, 0)
                    )

    # Write and display frame
    out.write(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()