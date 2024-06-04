import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# YOLO model initialization
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names to identify
classNames = ["car"]  # Only interested in cars

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
if not cap.isOpened():
    print(f"Error: Could not open video capture device.")
    exit()

# Load the mask image
mask_path = "mask.png"  # Change this to the correct path of your mask image
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Error: Could not load mask image from path: {mask_path}")
    exit()

# Tracking
tracker = Sort()

identified_count = 0  # Count of identified cars
unidentified_count = 0  # Count of unidentified objects

# Track which IDs have already been counted
counted_ids = set()

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read a frame from the camera")
        break

    frame_height, frame_width = img.shape[:2]

    # Resize the mask to match the frame dimensions
    mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    # Ensure the mask is of type CV_8U
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Define the barrier position
    barrier_position = frame_height // 2  # Position of the red line barrier

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
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with detections
    resultsTracker = tracker.update(detections)

    # Draw the red line barrier
    cv2.line(masked_frame, (0, barrier_position), (frame_width, barrier_position), (0, 0, 255), 2)

    # Draw bounding boxes and classify objects
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        if currentClass == "car":
            cvzone.cornerRect(masked_frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(masked_frame, f'car {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)
            if cy > barrier_position and id not in counted_ids:
                identified_count += 1
                counted_ids.add(id)
        else:
            cvzone.cornerRect(masked_frame, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 255))
            cvzone.putTextRect(masked_frame, 'unidentified', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)
            if id not in counted_ids:
                unidentified_count += 1
                counted_ids.add(id)

        cv2.circle(masked_frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Display total counts of identified and unidentified objects
    cvzone.putTextRect(masked_frame, f'Car Count: {identified_count}', (50, 50), scale=2, thickness=2, offset=10)
    cvzone.putTextRect(masked_frame, f'Unidentified Count: {unidentified_count}', (50, 100), scale=2, thickness=2, offset=10)

    # Show the frame
    cv2.imshow("Video", masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
