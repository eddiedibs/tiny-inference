#!/home/edd1e/projs/other/work_challenges/aisee_task/venv/bin/python3

from ultralytics import YOLO
import cv2
import sys

def convert_to_onnx(model_path, onnx_path):
    """
    Convert a YOLO model from .pt to ONNX format.
    
    Args:
        model_path (str): Path to the .pt model file.
        onnx_path (str): Path to save the ONNX model file.
    """
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=640)  # Export to ONNX format
    print(f"Model converted to ONNX and saved to {onnx_path}")

def run_inference(model_path, image_path, output_path):
    """
    Run inference on an image using the model.
    
    Args:
        model_path (str): Path to the model file (.pt or .onnx).
        image_path (str): Path to the input image file.
        output_path (str): Path to save the output image with bounding boxes.
    """
    # Load the model
    model = YOLO(model_path, task="detect")  # Explicitly set the task to 'detect'

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Perform prediction
    results = model(image)

    # Display and save the results
    for result in results:
        # Plot the bounding boxes on the image
        result_plotted = result.plot()
        
        # Save the image with bounding boxes
        cv2.imwrite(output_path, result_plotted)
        print(f"Output image saved to {output_path}")

        # Print the detected objects and their confidence scores
        boxes = result.boxes  # Bounding boxes
        masks = result.masks  # Masks (if any)
        keypoints = result.keypoints  # Keypoints (if any)
        probs = result.probs  # Class probabilities (if any)
        
        # Print detected objects
        for box in boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")

def main():
    helpmsg = """
    USAGE: main.py <command> <params> [--onnx <PATH>]
    
    COMMANDS:
        gen-model <model.pt> [--onnx <output.onnx>]  Generates ONNX file from .pt
        inf <model.pt or model.onnx> <image.png>     Processes Inference

    OPTIONS:
        --onnx <PATH>  Convert the model to ONNX format and save it to the specified path.
    """

    if len(sys.argv) < 2:
        print(helpmsg)
        return

    command = sys.argv[1]

    if command == "gen-model":
        if len(sys.argv) < 3:
            print("Error: Missing model path for gen-model command.")
            print(helpmsg)
            return

        model_path = sys.argv[2]
        onnx_path = None

        # Check if the --onnx flag is provided
        if '--onnx' in sys.argv:
            onnx_index = sys.argv.index('--onnx')
            if onnx_index + 1 < len(sys.argv):
                onnx_path = sys.argv[onnx_index + 1]
            else:
                print("Error: --onnx flag requires an output path.")
                return

        # Default ONNX path if not provided
        if onnx_path is None:
            onnx_path = model_path.replace(".pt", ".onnx")

        # Convert to ONNX
        convert_to_onnx(model_path, onnx_path)

    elif command == "inf":
        if len(sys.argv) < 4:
            print("Error: Missing model path or image path for inf command.")
            print(helpmsg)
            return

        model_path = sys.argv[2]
        image_path = sys.argv[3]

        # Run inference
        output_path = "./output.png"
        run_inference(model_path, image_path, output_path)

    else:
        print(f"Error: Unknown command '{command}'.")
        print(helpmsg)

if __name__ == "__main__":
    main()