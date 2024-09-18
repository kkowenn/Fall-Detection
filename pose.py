import cv2
import cvzone
import math
from ultralytics import YOLO
import mediapipe as mp

# Initialize YOLO and MediaPipe Pose
cap = cv2.VideoCapture('fall.mp4')
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

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
                # Draw bounding box for the person
                cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Extract the region of interest (ROI) for pose estimation
                person_roi = frame[y1:y2, x1:x2]

                # Convert the ROI to RGB for MediaPipe
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                # Pose estimation using MediaPipe
                result_pose = pose.process(rgb_roi)

                if result_pose.pose_landmarks:
                    # Extract landmarks for key points such as shoulders, hips, knees
                    landmarks = result_pose.pose_landmarks.landmark

                    # Get coordinates for shoulders and hips
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                    # Calculate the angles between shoulders and hips
                    shoulder_angle = abs(left_shoulder.y - right_shoulder.y)
                    hip_angle = abs(left_hip.y - right_hip.y)

                    # If the person is leaning forward or has fallen
                    if shoulder_angle < 0.1 and hip_angle < 0.1:
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    else:
                        cvzone.putTextRect(frame, 'No Fall', [x1, y1 - 40], thickness=2, scale=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
