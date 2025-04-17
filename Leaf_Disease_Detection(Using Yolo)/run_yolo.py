import cv2
from ultralytics import YOLO
import os, json

# Load labels from JSON

labels = {}
with open('./class_to_index_train.json', 'r') as f:
    labels = json.load(f)

# Interchange key:value pairs to value:key pairs
labels = {v: k for k, v in labels.items()}

# Load YOLO model
#Loads the best-trained YOLO model from the specified path.
#This model is used to make predictions on test images.

model = YOLO("./runs/detect/train/weights/best.pt")

# Load images from the directory
image_dir = "./data/images/test/"
image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPG')]
image_files.sort()

# Initialize variables for navigation
current_index = 0
total_images = len(image_files)

def display_image(index):
    img_path = os.path.join(image_dir, image_files[index])
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    results = model.predict(source=img, conf=0.8, verbose=1)
    
    for r in results:
        for box in r.boxes.data:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_index = int(box[5])
            label_name = labels.get(label_index, 'Unknown')
            # print(label_name)
            # Adjust text position to ensure visibility (move it just above the box, but clamp to image bounds)
            text_y = max(20, y_min - 10)  # Prevent text from going above y=0
            cv2.putText(img, label_name, (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Thickness reduced to 2
    
    # Create a resizable window and set initial size
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 680, 420)  # Set initial size (width, height)
    cv2.imshow('Image', img)

# Display the first image
display_image(current_index)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_index = (current_index + 1) % total_images
        display_image(current_index)
    elif key == ord('p'):
        current_index = (current_index - 1) % total_images
        display_image(current_index)

cv2.destroyAllWindows()