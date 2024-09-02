# Object Detection with YOLO

This repository contains a Python project for object detection using YOLO (You Only Look Once) with OpenCV. The project detects objects in images using the YOLOv3 model and saves the annotated images to a specified output directory.

## Project Structure

- `data/` - Contains YOLOv3 weights, configuration, and class names files.
  - `yolov3.weights` - Pre-trained weights for YOLOv3.
  - `yolov3.cfg` - YOLOv3 configuration file.
  - `coco.names` - Class names used by the YOLOv3 model.
- `images/`
  - `input/` - Directory containing input images for detection.
  - `output/` - Directory where annotated images will be saved.
- `object_detection.py` - Python script for running object detection.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Numpy (`numpy`)

Install the required Python packages:
pip install -r requirements.txt

Execute the object_detection.py script to start object detection:
python object_detection.py
