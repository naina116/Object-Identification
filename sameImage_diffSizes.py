import numpy as np
from ultralytics import YOLO
import cv2

# Initialize the YOLO model
model = YOLO("yolov8l.pt")  # Path to YOLOv8 weights

# Class names to identify (you can customize this list based on your use case)
classNames = model.names  # Using class names from the loaded YOLO model

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from path: {image_path}")
        return

    # Detect objects using YOLO
    results = model(img)

    detections = []
    for result in results:
        for box in result.boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = float(box.conf[0])
            # Class Name
            cls = int(box.cls[0])

            # Check if the class index is within the bounds of classNames list
            if cls < len(classNames):
                currentClass = classNames[cls]
            else:
                currentClass = "unknown"

            # Append the detection to the list if the confidence is above 0.3
            if conf > 0.3:
                detections.append((x1, y1, x2, y2, conf, currentClass, w, h))

    # Draw bounding boxes and classify objects
    for detection in detections:
        x1, y1, x2, y2, conf, currentClass, w, h = detection
        color = (255, 0, 255) if currentClass != "unknown" else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{currentClass} ({w}x{h})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the processed image
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
process_image("images/spoonsSize.jpg")  # Replace with the path to your image
