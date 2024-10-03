import os
import shutil
import numpy as np
from ultralytics import YOLO
from scipy.io import loadmat
from PIL import Image
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
base_path = 'stanford_dog'
images_path = os.path.join(base_path, 'Images')
annotations_path = os.path.join(base_path, 'Annotations')
output_path = os.path.join(base_path, 'yolo_dataset')

# Create YOLO dataset structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)


def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"0 {x_center} {y_center} {width} {height}"


# Load train and test splits
train_list = loadmat(os.path.join(annotations_path, 'train_list.mat'))['annotation_list']
test_list = loadmat(os.path.join(annotations_path, 'test_list.mat'))['annotation_list']

# Process train and validation data
for dataset, split in [(train_list, 'train'), (test_list, 'val')]:
    for file_path_array in tqdm(dataset, desc=f"Processing {split} data"):
        file_path = file_path_array[0][0] if isinstance(file_path_array, np.ndarray) else file_path_array

        if not isinstance(file_path, str):
            logging.warning(f"Unexpected file path type: {type(file_path)}")
            continue

        file_name = os.path.basename(file_path)

        # Copy image
        src_path = os.path.join(images_path, file_name)
        dst_path = os.path.join(output_path, 'images', split, file_name)

        if not os.path.exists(src_path):
            logging.warning(f"Source file not found: {src_path}")
            continue

        shutil.copy(src_path, dst_path)

        # Load annotation
        annotation_file = file_path.replace('.jpg', '')
        annotation_path = os.path.join(annotations_path, annotation_file)
        if not os.path.exists(annotation_path):
            logging.warning(f"Annotation file not found: {annotation_path}")
            continue

        annotation = loadmat(annotation_path)

        # Get image size
        with Image.open(src_path) as img:
            img_width, img_height = img.size

        # Convert bounding box to YOLO format
        try:
            bbox = annotation['annotation']['bndbox'][0][0][0]
            yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)

            # Write YOLO format annotation
            label_path = os.path.join(output_path, 'labels', split, file_name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(yolo_bbox)
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {str(e)}")

# Create dataset.yaml
dataset_yaml = f"""
path: {output_path}
train: images/train
val: images/val

nc: 1
names: ['dog']
"""

with open(os.path.join(output_path, 'dataset.yaml'), 'w') as f:
    f.write(dataset_yaml)

logging.info("Dataset preparation completed.")

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Fine-tune the model
results = model.train(
    data=os.path.join(output_path, 'dataset.yaml'),
    #data='D:/Programming/YOLO/stanford_dog/yolo_dataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='stanford_dogs_finetune'
)

# Evaluate the model
metrics = model.val()
logging.info(f"mAP50-95: {metrics.box.map}")
logging.info(f"mAP50: {metrics.box.map50}")
logging.info(f"mAP75: {metrics.box.map75}")

# Perform inference on a test image
test_image = os.path.join(output_path, 'images', 'val', os.listdir(os.path.join(output_path, 'images', 'val'))[0])
results = model(test_image)

# Show the results
for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('stanford_dogs_result.jpg')

logging.info("Training and evaluation completed.")