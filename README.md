# YOLOv5 & YOLOv8 Object Detection

This repository demonstrates object detection using **YOLOv5** and **YOLOv8** models from Ultralytics. The scripts process images and videos to detect objects and display annotated outputs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Image Detection (YOLOv5)](#image-detection-yolov5)
  - [Video Processing (YOLOv5 & YOLOv8)](#video-processing-yolov5--yolov8)
- [Model Options](#model-options)
- [Dependencies](#dependencies)
- [Credits](#credits)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/shahinur-alam/YOLO.git
    ```

2. Install dependencies:

    ```bash
    pip install torch torchvision torchaudio ultralytics opencv-python numpy
    ```

## Usage

### Image Detection (YOLOv5)

1. Modify `image_path` in the `detect_objects` function.
2. Run the script:

    ```bash
    python yolov5_image_detection.py
    ```

### Video Processing (YOLOv5 & YOLOv8)

#### YOLOv5

1. Set the `video_source` to the path of the video or webcam.
2. Run the script:

    ```bash
    python yolov5_video_detection.py
    ```

#### YOLOv8

1. Set the `video_path` to the video file or webcam.
2. Run the script:

    ```bash
    python yolov8_video_detection.py
    ```

## Model Options

You can switch model sizes based on speed and accuracy:

- **YOLOv5**: `'yolov5n'`, `'yolov5s'`, `'yolov5m'`, `'yolov5l'`, `'yolov5x'`
- **YOLOv8**: `'yolov8n.pt'`, `'yolov8s.pt'`, `'yolov8m.pt'`, `'yolov8l.pt'`, `'yolov8x.pt'`

## Dependencies

- Python 3.x
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy

## Credits

- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
