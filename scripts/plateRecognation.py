import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont

# ========== Utility Functions ==========

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
    plate_text = plate_text.replace('О', 'O').replace('З', '3')  # Optional cleanup
    digits = ''.join(c for c in plate_text if c.isdigit())
    letters = ''.join(c for c in plate_text if c.isalpha())
    return (digits[:4] + letters[:3])[:7]

def get_most_common_plate(track_id, history):
    fixed_texts = []
    for t in history[track_id]:
        fixed = fix_plate_format(t)
        if len(fixed) == 7:
            fixed_texts.append(fixed)
    return Counter(fixed_texts).most_common(1)[0][0] if fixed_texts else ""

def centroid_distance(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.linalg.norm(np.array(c1) - np.array(c2))

# ========== Load Models ==========

plate_model = YOLO('../models/plate_model.pt')
char_model = YOLO('../models/best4.pt')

# ========== Load Video and Initialize ==========

cap = cv2.VideoCapture('../test_video3.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('output_tracked.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

track_history = defaultdict(list)
char_history = defaultdict(list)
next_track_id = 1

char_map = {i: str(i) for i in range(10)}
char_map.update({
    10: 'А', 11: 'Б', 12: 'В', 13: 'Г', 14: 'Д', 15: 'Е', 16: 'Ё', 17: 'Ж', 18: 'З', 19: 'И',
    20: 'Й', 21: 'К', 22: 'Л', 23: 'М', 24: 'Н', 25: 'О', 26: 'П', 27: 'Р', 28: 'С', 29: 'Т',
    30: 'У', 31: 'Ф', 32: 'Х', 33: 'Ц', 34: 'Ч', 35: 'Ш', 36: 'Щ', 37: 'Ъ', 38: 'Ы', 39: 'Ь',
    40: 'Э', 41: 'Ю', 42: 'Я', 43: 'Ө', 44: 'Ү'
})

# ========== Main Loop ==========

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    plate_results = plate_model(frame, conf=0.5)
    if not plate_results or not plate_results[0].boxes:
        out.write(frame)
        continue

    current_plates = []
    for r in plate_results:
        for box in r.boxes:
            current_plates.append(list(map(int, box.xyxy[0])))

    plate_to_track_id = {}
    for plate in current_plates:
        x1, y1, x2, y2 = plate
        matched, match_id = False, None
        min_distance = float('inf')

        for tid, history in track_history.items():
            if history:
                dist = centroid_distance(plate, history[-1])
                if dist < min_distance and dist < 50:
                    min_distance = dist
                    match_id = tid
                    matched = True

        if matched:
            track_history[match_id].append(plate)
            plate_to_track_id[tuple(plate)] = match_id
        else:
            track_id = next_track_id
            next_track_id += 1
            track_history[track_id].append(plate)
            plate_to_track_id[tuple(plate)] = track_id

    for plate, track_id in plate_to_track_id.items():
        x1, y1, x2, y2 = plate
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        try:
            # Double height to accommodate both one-line and two-line formats
            plate_img_resized = cv2.resize(plate_img, (640, 416))
        except Exception as e:
            print(f"Resize error: {e}")
            continue

        char_results = char_model(plate_img_resized, conf=0.25)
        chars = []

        for r in char_results:
            for box in r.boxes:
                cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                cls = int(box.cls[0])
                chars.append((cx, char_map.get(cls, '?')))

        chars.sort(key=lambda x: x[0])
        plate_text = ''.join(c[1] for c in chars)

        if plate_text:
            char_history[track_id].append(plate_text)

        final_plate = get_most_common_plate(track_id, char_history)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        frame = draw_mongolian_text_on_image(
            image=frame,
            text=f'Plate: {final_plate}',
            position=(x1, max(0, y1 - 50)),
            font_path="../Roboto/static/Roboto-Regular.ttf",
            font_size=36,
            color=(0, 255, 0)
        )

    
    out.write(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()