import cv2
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply, Reshape, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Custom AttentionBlock
class AttentionBlock(Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channel_attention = Sequential([
            GlobalAveragePooling2D(),
            Dense(input_shape[-1] // 8, activation='relu'),
            Dense(input_shape[-1], activation='sigmoid'),
            Reshape((1, 1, input_shape[-1]))
        ])
        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs):
        attention_weights = self.channel_attention(inputs)
        return Multiply()([inputs, attention_weights])
    
    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load models
yolo_model = YOLO("runs/detect/train/weights/best.pt")
hybrid_model = tf.keras.models.load_model('hybrid_cnn_with_attention.keras', custom_objects={'AttentionBlock': AttentionBlock})
# Load YOLO labels (adjust path as needed)
with open('class_to_index_train.json', 'r') as f:
    import json
    yolo_labels = json.load(f)
yolo_labels = {v: k for k, v in yolo_labels.items()}

# Hybrid CNN labels (replace with your 38 class names if available)
hybrid_labels = [f"Disease_{i}" for i in range(38)]  # Placeholder

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Image directory
image_dir = "data/images/test/test"
image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPG')]
image_files.sort()

# Preprocess for Hybrid CNN
def preprocess_for_hybrid(image_path:str):
    # Preprocess the image for the model
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    return input_arr

# Display function
def display_image(index):
    img_path = os.path.join(image_dir, image_files[index])
    img = cv2.imread(img_path)

    # YOLOv8 detection
    results = yolo_model.predict(source=img, conf=0.6, verbose=False)
    
    for r in results:
        for box in r.boxes.data:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            yolo_label_idx = int(box[5])
            yolo_label = yolo_labels.get(yolo_label_idx, 'Unknown')

            # Draw YOLO box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text_y = max(20, y_min - 10)
            cv2.putText(img, f"YOLO: {yolo_label}", (x_min, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop and classify with Hybrid CNN
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size > 0:

                img_array = preprocess_for_hybrid(img_path)
                hybrid_pred = hybrid_model.predict(img_array, verbose=0)
                hybrid_label_idx = np.argmax(hybrid_pred)
                hybrid_label = class_names[hybrid_label_idx]
                cv2.putText(img, f"CNN: {hybrid_label}", (x_min, text_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 680, 420)
    cv2.imshow('Image', img)

# Navigation
current_index = 0
total_images = len(image_files)
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

