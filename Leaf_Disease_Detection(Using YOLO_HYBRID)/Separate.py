import cv2
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply, Reshape, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
import json

# Custom AttentionBlock Layer
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

# YOLO Detection Interface
class YOLODetector:
    def __init__(self, model_path, labels_path):
        self.model = YOLO(model_path)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.labels = {v: k for k, v in self.labels.items()}

    def detect(self, image, conf_threshold=0.8):
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes.data:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                label_idx = int(box[5])
                label = self.labels.get(label_idx, 'Unknown')
                confidence = float(box[4])
                detections.append({
                    'box': (x_min, y_min, x_max, y_max),
                    'label': label,
                    'confidence': confidence
                })
        return detections

# Hybrid CNN Classification Interface
class HybridCNNClassifier:
    def __init__(self, model_path, class_names):
        self.model = load_model(model_path, custom_objects={'AttentionBlock': AttentionBlock})
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        return input_arr

    def classify(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        label_idx = np.argmax(predictions)
        label = self.class_names[label_idx]
        confidence = float(predictions[0][label_idx])
        return label, confidence

# Main Integrated Interface with Separate Displays
class PlantDiseaseAnalyzer:
    def __init__(self, yolo_detector:YOLODetector, hybrid_classifier, image_dir):
        self.yolo_detector = yolo_detector
        self.hybrid_classifier = hybrid_classifier
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.JPG')]
        self.image_files.sort()

    def display_yolo(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        img = cv2.imread(img_path)
        
        # YOLO Detection
        detections = self.yolo_detector.detect(img, 0.6)
        for detection in detections:
            x_min, y_min, x_max, y_max = detection['box']
            yolo_label = detection['label']
            yolo_conf = detection['confidence']

            # Draw YOLO box and label with confidence
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text_y = max(15, y_min - 5)  # Position above box
            label_text = f"YOLO: {yolo_label} ({yolo_conf:.2f})"
            cv2.putText(img, label_text, (x_min, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)  # Font scale reduced to 0.35

        # Display YOLO window
        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection', 680, 420)
        cv2.imshow('YOLO Detection', img)

    def display_hybrid_cnn(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        img = cv2.imread(img_path)
        
        # Hybrid CNN Classification
        hybrid_label, hybrid_conf = self.hybrid_classifier.classify(img_path)
        label_text = f"CNN: {hybrid_label} ({hybrid_conf:.2f})"
        cv2.putText(img, label_text, (10, 20),  # Position near top-left
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)  # Font scale reduced to 0.35

        # Display Hybrid CNN window
        cv2.namedWindow('Hybrid CNN Classification', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hybrid CNN Classification', 680, 420)
        cv2.imshow('Hybrid CNN Classification', img)

    def run(self):
        current_index = 0
        total_images = len(self.image_files)
        
        # Initial display
        self.display_yolo(current_index)
        self.display_hybrid_cnn(current_index)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_index = (current_index + 1) % total_images
                self.display_yolo(current_index)
                self.display_hybrid_cnn(current_index)
            elif key == ord('p'):
                current_index = (current_index - 1) % total_images
                self.display_yolo(current_index)
                self.display_hybrid_cnn(current_index)
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Class names for Hybrid CNN
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

    # Initialize components
    yolo_detector = YOLODetector("runs/detect/train/weights/best.pt", "class_to_index_train.json")
    hybrid_classifier = HybridCNNClassifier("hybrid_cnn_with_attention.keras", class_names)
    analyzer = PlantDiseaseAnalyzer(yolo_detector, hybrid_classifier, "data/images/test/test")

    # Run the integrated system
    analyzer.run()