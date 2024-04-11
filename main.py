import os
import cv2
import sys
from unittest.mock import DEFAULT

# Constants
CONF_THRESHOLD_1 = 0.5
CONF_THRESHOLD_2 = 0.3
CLASS_FILE_PATH = 'files/thing.names'
MODEL_FILE_PATH = 'files/frozen_inference_graph.pb'
CONFIG_FILE_PATH = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
OUTPUT_IMAGE_PATH = '/images/ion_1.jpg'
OUTPUT_WINDOW_NAME = 'ION'

def banner():
    print("""
    \33[96m
    ██╗ ██████╗ ███╗   ██╗
    ██║██╔═══██╗████╗  ██║
    ██║██║   ██║██╔██╗ ██║
    ██║██║   ██║██║╚██╗██║
    ██║╚██████╔╝██║ ╚████║
    ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    \33[92m
    github:amine10a
    fb:emine.ardhaoui
    """)

def select_option():
    banner()
    print("""
    [1] Select image objects
    [2] Select deep image objects
    [3] Exit
    """)
    while True:
        try:
            choice = int(input("Enter your choice: "))
            if choice in [1, 2, 3]:
                return choice
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def process_image(img, conf_threshold):
    h, w, c = img.shape
    print(f"Image dimensions - Height: {h}, Width: {w}, Channels: {c}")

    if h > 1200 and w > 1200:
        img = cv2.resize(img, (w // 10, h // 10))

    # Load class names from file
    with open(CLASS_FILE_PATH, 'r') as f:
        class_names = f.read().splitlines()

    # Load model
    net = cv2.dnn_DetectionModel(MODEL_FILE_PATH, CONFIG_FILE_PATH)
    net.setInputSize(320, 240)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Detect objects
    class_ids, confidences, bboxes = net.detect(img, conf_threshold)
    print(f"Detected classes: {class_ids}, Bounding boxes: {bboxes}")

    for class_id, confidence, bbox in zip(class_ids.flatten(), confidences.flatten(), bboxes):
        cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=3)
        cv2.putText(img, class_names[class_id - 1], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    cv2.imshow(OUTPUT_WINDOW_NAME, img)
    cv2.waitKey(0)

    # Save image
    if cv2.imwrite(OUTPUT_IMAGE_PATH, img):
        print("Image saved successfully.")
    else:
        print("Failed to save the image.")

def main():
    choice = select_option()

    if choice in [1, 2]:
        image_path = input("Enter image path: ")
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Failed to load image from path: {image_path}")
            return
        
        conf_threshold = CONF_THRESHOLD_1 if choice == 1 else CONF_THRESHOLD_2
        process_image(img, conf_threshold)
    
    elif choice == 3:
        print("Exiting the program.")
        sys.exit()

if __name__ == '__main__':
    main()
