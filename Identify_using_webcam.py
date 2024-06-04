# Import necessary libraries
from ultralytics import YOLO
import cv2
import time

# Initialize YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","earbuds","key"]

# Initialize variables for FPS calculation
prev_frame_time = 0

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Main loop for video processing
while True:
    # Read frame from the webcam
    success, img = cap.read()

    # Perform object detection
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates and class
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]), 2)

            # Draw bounding box and class label
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Ensure class index is within the range of classNames list
            if cls < len(classNames):
                class_label = classNames[cls]
            else:
                class_label = "Unknown"
                print("Invalid class index:", cls)
            cv2.putText(img, f'{classNames[cls]} {conf}', (int(x1), max(35, int(y1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Display processed frame
    cv2.imshow("Image", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

