# tiny-inference - YOLO Model Conversion & Inference

## Overview
This project provides scripts to convert a YOLO model from .pt format to ONNX and perform inference on images using either .pt or .onnx models.

## Requirements
Make sure you have the following dependencies installed:
pip install ultralytics opencv-python

## Usage
### The script supports two main functionalities:
1. Model Conversion: Convert a .pt YOLO model to ONNX format.
2. Inference: Run object detection on an image.

## Commands
### Generate ONNX Model
`python main.py gen-model <model.pt> [--onnx <output.onnx>]`

`<model.pt>`: Path to the .pt model.
`--onnx <output.onnx>` (optional): Path to save the converted ONNX model. If not provided, it defaults to `<model.onnx>`.

### Example:
`python main.py gen-model yolov8.pt --onnx yolov8.onnx`

## Run Inference
`python main.py inf <model.pt or model.onnx> <image.png>`

`<model.pt or model.onnx>`: Path to the model file (either .pt or .onnx).
`<image.png>`: Path to the image on which inference will be performed.

### Example:
`python main.py inf yolov8.pt input_image.jpg`

### Output
- The output image with bounding boxes is saved as output.png.
- The script prints detected objects, their confidence scores, and bounding box coordinates.

### Notes
- Ensure the image resolution matches the model's expected input size.
- The script supports YOLO models trained for object detection (task="detect").
