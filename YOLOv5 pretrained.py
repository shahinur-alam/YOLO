# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:35:54 2024

@author: shahinur
"""

import cv2
import torch
import numpy as np


# Check if CUDA is available and set the device
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f"Using device: {device}")


# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Set model to evaluation mode
model.eval()

#For Image
'''
def detect_objects(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()

    print(f"Number of detections: {len(detections)}")

    # Draw bounding boxes and labels
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        if conf > 0.5:  # Confidence threshold
            print(f"Detected: {model.names[int(class_id)]} with confidence {conf:.2f}")
            label = f"{model.names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLOv5 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")


# Example usage
image_path = "demo.jpg"
detect_objects(image_path)
'''

#For video
def process_video(video_source):
    # Open video capture
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))  # Webcam
    else:
        cap = cv2.VideoCapture(video_source)  # Video file

    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    out = cv2.VideoWriter('output/output_video_v5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(frame_rgb)

        # Extract detections
        detections = results.xyxy[0].cpu().numpy()

        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            if conf > 0.5:  # Confidence threshold
                label = f"{model.names[int(class_id)]}: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv5 Detection", frame)

        # Write the frame to output video
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete. Output saved as 'output_video.mp4'")


# Example usage
video_source = "input/video.mp4"  # Or use "0" for webcam
process_video(video_source)