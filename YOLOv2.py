# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:25:52 2024

@author: shahinur
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as ops

# Define YOLOv2 Model (same as before)
class YOLOv2(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = self.num_anchors * (5 + self.num_classes)  # (x, y, w, h, confidence) + class_probs

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, self.output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

# Anchor Boxes (Example Values)
anchors = [
    [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]
]

# Preprocessing the input frame
def preprocess_frame(frame, input_size=416):
    frame = cv2.resize(frame, (input_size, input_size))
    frame = frame / 255.0  # Normalize
    frame = np.transpose(frame, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return torch.Tensor(frame)

# Decoding YOLOv2 output (you need to write this function as per your model's needs)
def decode_predictions(output, anchors, num_classes):
    # You will need to implement this to extract the bounding boxes, class scores, and confidences.
    pass

# Draw bounding boxes on the frame
def draw_boxes(frame, boxes, confidences, class_labels):
    for (box, conf, label) in zip(boxes, confidences, class_labels):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Main Function for Video Processing
def process_video(video_path, model, anchors, num_classes=20):
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Perform forward pass through YOLOv2 model
        model.eval()
        with torch.no_grad():
            output = model(input_frame)

        # Decode the output
        boxes, confidences, class_labels = decode_predictions(output, anchors, num_classes)

        # Draw boxes on the frame
        draw_boxes(frame, boxes, confidences, class_labels)

        # Display the frame with bounding boxes
        cv2.imshow("YOLOv2 Video", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Instantiate the model
model = YOLOv2(num_classes=20, num_anchors=5)  # Example for Pascal VOC dataset

# Process video file
process_video("video.mp4", model, anchors, num_classes=20)
