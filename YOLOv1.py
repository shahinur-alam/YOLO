# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 21:55:05 2024

@author: shahinur
"""

import cv2
import torch
import torch.nn as nn
import numpy as np

# YOLOv1 Model
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):  # S = Grid size, B = Number of boxes, C = Number of classes
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))  # Output shape SxSx(B*5 + C)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)  # Reshape output


def preprocess_image(image, input_size=448):
    """Preprocess input image for YOLOv1."""
    image = cv2.resize(image, (input_size, input_size))
    image = image / 255.0  # Normalize pixel values
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.Tensor(image)


def decode_output(output, S=7, B=2, C=20):
    """Decode the YOLOv1 output."""
    grid_size = 1.0 / S
    boxes = []
    class_probs = []
    for i in range(S):
        for j in range(S):
            cell = output[i, j, :]
            for b in range(B):
                box = cell[b * 5:(b + 1) * 5]  # Extract (x, y, w, h, confidence)
                confidence = box[4]
                if confidence > 0.5:  # Threshold
                    x = (box[0] + j) * grid_size
                    y = (box[1] + i) * grid_size
                    w = box[2] * grid_size
                    h = box[3] * grid_size
                    boxes.append([x, y, w, h, confidence])
                    class_probs.append(cell[B * 5:])
    return boxes, class_probs


def draw_boxes(img, boxes):
    for box in boxes:
        x, y, w, h, confidence = box
        left = int((x - w / 2) * img.shape[1])
        top = int((y - h / 2) * img.shape[0])
        right = int((x + w / 2) * img.shape[1])
        bottom = int((y + h / 2) * img.shape[0])
        
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"Confidence: {confidence:.2f}"
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Load YOLOv1 model
model = YOLOv1(S=7, B=2, C=20)  # Customize S, B, and C for your case

# You would normally load pre-trained weights, but for simplicity, this is skipped here.
# model.load_state_dict(torch.load("yolov1_weights.pth"))

# Set the model to evaluation mode
model.eval()

# Open video file or capture from webcam
cap = cv2.VideoCapture("video.mp4", cv2.CAP_FFMPEG)  # Change to 0 for webcam

#cap = cv2.VideoCapture(0, cv2.CAP_ANY)  # Use default backend, should resolve issues with GStreamer


while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the video frame
    input_frame = preprocess_image(frame)

    # Run YOLOv1 forward pass
    with torch.no_grad():
        output = model(input_frame)

    # Decode the output to get boxes
    boxes, class_probs = decode_output(output[0])

    # Draw the detected boxes on the frame
    draw_boxes(frame, boxes)

    # Display the frame with bounding boxes
    cv2.imshow("YOLOv1 Video", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
# Release the video capture object
cap.release()

cv2.destroyAllWindows()

