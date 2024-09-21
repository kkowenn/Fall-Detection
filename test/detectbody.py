import cv2
import cvzone
import math
from ultralytics import YOLO
import mediapipe as mp
import ssl

# Bypass SSL certificate verification for local testing (temporary fix for SSL errors)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize YOLO and MediaPipe Pose
cap = cv2.VideoCapture('fall.mp4')
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)  # Use higher complexity for better accuracy

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Threshold for floor detection (adjust based on camera position and frame size)
FLOOR_THRESHOLD_Y = 650  # Assuming a floor threshold in pixel coordinates
fall_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Maintain the aspect ratio of the frame
    frame = cv2.resize(frame, (980, 740), interpolation=cv2.INTER_AREA)

    # YOLO person detection
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
                # Adjust the bounding box size slightly if needed
                margin = 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                # Draw bounding box for the person
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Instead of just extracting the ROI, apply pose estimation on the entire frame for better alignment
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Pose estimation using MediaPipe
                result_pose = pose.process(rgb_frame)

                if result_pose.pose_landmarks:
                    # Draw skeleton and pose landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        result_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Extract landmarks for key points such as head, shoulders, hips, knees
                    landmarks = result_pose.pose_landmarks.landmark

                    # Store the y-coordinate of key body parts
                    body_parts = {
                        'head': landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0],
                        'left_shoulder': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0],
                        'right_shoulder': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0],
                        'left_hip': landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0],
                        'right_hip': landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0],
                        'left_knee': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0],
                        'right_knee': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0],
                        'left_ankle': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0],
                        'right_ankle': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]
                    }

                    # Detect fall by monitoring the y-coordinate change
                    if not fall_detected:
                        # Check if the person is falling (e.g., sudden drop in the y-position of the hips or shoulders)
                        if body_parts['head'] > FLOOR_THRESHOLD_Y and body_parts['left_hip'] > FLOOR_THRESHOLD_Y:
                            fall_detected = True
                            cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    else:
                        # After the fall is detected, check for impact
                        hitting_part = min(body_parts, key=body_parts.get)

                        # If the part is near the floor threshold, we detect the first impact
                        if body_parts[hitting_part] > FLOOR_THRESHOLD_Y:
                            cvzone.putTextRect(frame, f'{hitting_part} hit the floor first', [x1, y1 - 40], thickness=2, scale=2)
                        else:
                            cvzone.putTextRect(frame, 'Impact Detected', [x1, y1 - 40], thickness=2, scale=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
