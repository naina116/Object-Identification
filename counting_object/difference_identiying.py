import cv2
import cvzone
import numpy as np
import math
from ultralytics import YOLO
from sort import *

# Path to the input video and mask image
video_path = "cars.mp4"  # Change this to the correct path of your video
mask_path = "mask.png"   # Change this to the correct path of your mask image

# YOLO model initialization
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names to identify
classNames = ["car", "bike", "truck"]  # Classes to identify

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Load the mask image
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Error: Could not load mask image from path: {mask_path}")
    exit()

# Get the frame dimensions from the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the video")
    exit()
frame_height, frame_width = frame.shape[:2]

# Resize the mask to match the frame dimensions
mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

# Ensure the mask is of type CV_8U
if mask.dtype != np.uint8:
    mask = mask.astype(np.uint8)

# Tracking
tracker = Sort()

# Define the barrier position
barrier_position = frame_height // 2  # Position of the red line barrier

while True:
    ret, img = cap.read()
    if not ret:
        break  # Exit the loop if there are no more frames

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(img, img, mask=mask)

    # Detect objects using YOLO
    results = model(masked_frame, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Check if the class index is within the bounds of classNames list
            if cls < len(classNames):
                currentClass = classNames[cls]
            else:
                currentClass = "unidentified"

            # Create a detection array if the confidence is above 0.3
            if conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf, currentClass])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with detections
    resultsTracker = tracker.update(detections)

    # Draw the red line barrier
    cv2.line(masked_frame, (0, barrier_position), (frame_width, barrier_position), (0, 0, 255), 2)

    # Track which IDs have already crossed the barrier
    crossed_ids = set()

    # Draw bounding boxes and classify objects
    for result in resultsTracker:
        x1, y1, x2, y2, id, currentClass = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        if cy > barrier_position and id not in crossed_ids:
            cvzone.cornerRect(masked_frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(masked_frame, f'{currentClass}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)
            crossed_ids.add(id)

        cv2.circle(masked_frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Show the frame
    cv2.imshow("Video", masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

