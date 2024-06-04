import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Path to the input image
image_path = "1.jpg"

# YOLO model initialization
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names (including "person" for this example)
classNames = ["person"]

# Load the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Image not found or unable to load from path: {image_path}")
    exit()

# Tracking
tracker = Sort()

totalCount = []

# Detect objects using YOLO
results = model(img, stream=True)

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

        # Check if the detected object is a person and the confidence is above 0.3
        if currentClass == "person" and conf > 0.3:
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

# Update the tracker with detections
resultsTracker = tracker.update(detections)

# Draw bounding boxes and count people
for result in resultsTracker:
    x1, y1, x2, y2, id = result
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2

    if currentClass != "unidentified":
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{currentClass} {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
    else:
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 255))
        cvzone.putTextRect(img, 'unidentified', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Check if person is detected
    if id not in totalCount:
        totalCount.append(id)

# Display total count
cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50), scale=2, thickness=2, offset=10)

# Show the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
