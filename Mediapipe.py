import cv2
import cvzone
import math
from ultralytics import YOLO
import mediapipe as mp
import os
import numpy as np

# Create a save directory if it doesn't exist
save_folder = 'save'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture('fall.mp4')

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Couldn't read video stream from file 'fall.mp4'")
    exit()

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Try to load class names from 'classes.txt', handle if the file is not found
classnames = []
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except FileNotFoundError:
    print("Warning: 'classes.txt' not found. Using default class names.")
    classnames = ['person']  # Default to 'person' if no class names are available

frame_count = 0  # Initialize a frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    # Create a blank image (black background)
    skeleton_image = np.zeros_like(frame)

    # Get results from YOLO model
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect] if class_detect < len(classnames) else 'Unknown'
            conf = math.ceil(confidence * 100)

            # Implement fall detection using the coordinates x1, y1, x2, y2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                # Convert to RGB for Mediapipe
                rgb_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

                # Get pose landmarks
                result_pose = pose.process(rgb_crop)

                # Draw a line using pose landmarks on the skeleton_image (blank image)
                if result_pose.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        skeleton_image[y1:y2, x1:x2], result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Check for fall detection based on the bounding box ratio
                if threshold < 0:
                    cvzone.putTextRect(skeleton_image, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)

            else:
                pass

    cv2.imshow('frame', skeleton_image)

    # Save each frame in the save folder with a unique name
    save_path = os.path.join(save_folder, f'skeleton_frame_{frame_count}.jpg')
    cv2.imwrite(save_path, skeleton_image)
    frame_count += 1  # Increment the frame counter

    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
