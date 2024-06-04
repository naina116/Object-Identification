import cv2

# Load the ONNX model
net = cv2.dnn.readNetFromONNX('yolov5s.onnx')

# Load the video capture device (e.g., webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True)

    # Set the input blob
    net.setInput(blob)

    # Forward pass to get the detection results
    detections = net.forward()

    # Process the detections (draw bounding boxes, labels, etc.)
    # ...

    # Display the processed frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
