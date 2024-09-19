import cv2
import cvzone
import math
from ultralytics import YOLO
import os

# Initialize YOLO
cap = cv2.VideoCapture('fall.mp4')
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

output_dir = 'save'  # Directory to save fall images
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fall_detected = False
fall_image_saved = False  # To ensure the image is saved only once after the fall is detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            if conf > 80 and class_detect == 'person':
                # Draw bounding box for the detected person
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Check if the person is lying down (falling position)
                height = y2 - y1
                width = x2 - x1
                # Logic: If height is less than width, it's likely a person lying down
                if height < width:
                    fall_detected = True
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    print(f"Fall detected at Frame: Height {height}, Width {width}")

                    if not fall_image_saved:
                        fall_path = os.path.join(output_dir, f"fall_image_{cv2.getTickCount()}.jpg")
                        cv2.imwrite(fall_path, frame)
                        fall_image_saved = True
                        print(f"Fall image saved at {fall_path}")
                else:
                    fall_detected = False

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
