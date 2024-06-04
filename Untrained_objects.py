from ultralytics import YOLO
import cv2
import numpy as np

# Initialize the YOLO model with pre-trained weights
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Define the specific classes you want to detect
target_classes = ["book", "person", "ketchup", "cup", "laptop","bottle","scissors","packet"]

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Detect objects in the image
    results = model(img)

    # Create a copy of the image to draw on
    output_img = img.copy()

    # Iterate through the detected objects and draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {confidence:.2f}'

            # Check if the detected class is one of the target classes
            if model.names[cls] in target_classes:
                # Draw the bounding box and label on the image
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the image for better visualization if necessary
    screen_res = 1280, 720
    scale_width = screen_res[0] / output_img.shape[1]
    scale_height = screen_res[1] / output_img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(output_img.shape[1] * scale)
    window_height = int(output_img.shape[0] * scale)

    # Display the image with detections
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detections', window_width, window_height)
    cv2.imshow('Detections', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your image file
image_path = 'images/ketchup.jpg'

# Process the image
process_image(image_path)
