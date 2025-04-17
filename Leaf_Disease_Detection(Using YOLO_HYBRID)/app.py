import streamlit as st
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Multiply, Reshape
from tensorflow.keras.models import Sequential
import json
import time
from PIL import Image
import io
from ultralytics import YOLO

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
    def __init__(self, model_path:str, labels_path:str):
        self.model = YOLO(model_path)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.labels = {v: k for k, v in self.labels.items()}

    def detect(self, image:cv2.typing.MatLike, conf_threshold=0.55):

        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        detections = []
        images = []

        # Convert PIL image to cv2 Image
        image = np.array(image)
        
        for r in results:
            for box in r.boxes.data:
                processed_img = image.copy()
                x_min, y_min, x_max, y_max = map(int, box[:4])
                label_idx = int(box[5])
                label = self.labels.get(label_idx, 'Unknown')
                confidence = float(box[4])
                cv2.rectangle(processed_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Adjust text position to ensure visibility (move it just above the box, but clamp to image bounds)
                text_y = max(20, y_min - 10)  # Prevent text from going above y=0
                cv2.putText(processed_img, label, (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Thickness reduced to 2
                images.append(processed_img)
                detections.append({
                    'box': (x_min, y_min, x_max, y_max),
                    'label': label,
                    'confidence': confidence
                })

        
        if(detections):
            max_index = max(enumerate(detections), key=lambda x: x[1]['confidence'])[0]
            image = images[max_index]
        return detections, image

# Hybrid CNN Classification Interface
class HybridCNNClassifier:
    def __init__(self, model_path, class_names):
        self.model = load_model(model_path, custom_objects={'AttentionBlock': AttentionBlock})
        self.class_names = class_names

    def preprocess_image(self, image):
        if isinstance(image, str):  # If path is provided
            img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
        else:  # If image array is provided
            img_array = tf.image.resize(image, (128, 128))
        
        input_arr = np.array([img_array])
        return input_arr

    def classify(self, image):
        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array, verbose=0)
        label_idx = np.argmax(predictions)
        label = self.class_names[label_idx]
        confidence = float(predictions[0][label_idx])
        return label, confidence, predictions

# Class names for the classifier
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]

# Helper function to convert uploaded file to OpenCV format
def image_prepocess_cnn(file):
    pil_image = Image.open(file)
    return np.array(pil_image)

def image_preprocess_yolo(file):
    return Image.open(file)

# Function to process image with both models
def process_image(yolo_detector:YOLODetector,image_yolo, hybrid_classifier,image_cnn ):
    # YOLO Detection
    detections,yolo_image = yolo_detector.detect(image_yolo)

    # Hybrid CNN Classification - using the full image
    hybrid_label, hybrid_conf, predictions = hybrid_classifier.classify(image_cnn)

    return yolo_image, detections, hybrid_label, hybrid_conf, predictions

# Initialize models
@st.cache_resource
def load_models():
    try:
        yolo_detector = YOLODetector("runs/detect/train/weights/best.pt", "class_to_index_train.json")
        hybrid_classifier = HybridCNNClassifier("hybrid_cnn_with_attention.keras", class_names)
        return yolo_detector, hybrid_classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Streamlit App
def main():
    st.set_page_config(page_title="Plant Disease Analyzer", layout="wide")

    # Sidebar
    st.sidebar.title("🌿 Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "About"])

    # Load models
    if app_mode == "Disease Recognition":
        with st.spinner("Loading models..."):
            yolo_detector, hybrid_classifier = load_models()
    
    # Main Page
    if app_mode == "Home":
        st.header("🌿 ADVANCED LEAF DISEASE RECOGNITION SYSTEM")
        
        # Try to load the image, but use a placeholder if not found
        try:
            image_path = "home_page.jpg"
            st.image(image_path, use_container_width=True)
        except:
            st.info("🖼 Home page image not found. Please add 'home_page.jpg' to your directory.")
        
        st.markdown(
            """
            ## Welcome to the Advanced Leaf Disease Recognition System! 🍃🔍
            
            Our system combines two powerful AI technologies to provide accurate plant disease detection:
            
            1. *Object Detection (YOLO)*: Precisely locates diseased areas on the leaf
            2. *Classification (Hybrid CNN)*: Identifies the specific disease affecting your plant
            
            ### 🌱 How It Works:
            1. *Upload an Image* - Go to the *Disease Recognition* page and upload a clear image of a plant leaf.
            2. *Dual AI Analysis* - The system analyzes your image using both YOLO object detection and our specialized Hybrid CNN with attention mechanism.
            3. *Detailed Results* - Get a comprehensive diagnosis showing both localized detection and overall classification.
            
            ### ✅ Why Choose Our Advanced System?
            - *Higher Accuracy* - Two complementary AI models working together
            - *Visual Explanation* - See exactly where the disease appears on the leaf
            - *Confident Diagnosis* - Get confidence scores for more reliable results
            
            🔍 *Ready to begin?* Head over to the *Disease Recognition* page and start diagnosing your plants today!
            """
        )
    
    # Recognition Page
    elif app_mode == "Disease Recognition":
        st.header("🌿 Advanced Disease Recognition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            test_image = st.file_uploader("📷 Upload a leaf image:", type=["jpg", "jpeg", "png"])
            
            analysis_method = st.radio(
                "Select Analysis Method:",
                ["Combined Analysis", "YOLO Detection Only", "Hybrid CNN Only"]
            )
            
            if test_image is not None:
                st.image(test_image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Disease"):
                    try:
                        with st.spinner("🍃 Analyzing image... This may take a moment!"):
                            # Convert uploaded file to CV2 format
                            image_cnn = image_prepocess_cnn(test_image)
                            image_yolo = image_preprocess_yolo(test_image)
                            
                            # Process with both models
                            result_image, detections, hybrid_label, hybrid_conf, predictions = process_image(
                                yolo_detector,image_yolo, hybrid_classifier,image_cnn
                            )
                            
                            # Store results in session state for display
                            st.session_state.result_image = result_image
                            st.session_state.detections = detections
                            st.session_state.hybrid_label = hybrid_label
                            st.session_state.hybrid_conf = hybrid_conf
                            st.session_state.predictions = predictions
                            st.session_state.analyzed = True
                            
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
        
        with col2:
            st.subheader("Analysis Results")
            
            if test_image is not None and 'analyzed' in st.session_state and st.session_state.analyzed:
                # Display results based on selected method
                if analysis_method == "Combined Analysis" or analysis_method == "YOLO Detection Only":
                    # Show YOLO results
                    st.subheader("YOLO Object Detection Results")
                    st.image(st.session_state.result_image, caption="Detection Results", use_container_width=True)
                    
                    if len(st.session_state.detections) > 0:
                        st.success(f"Found {len(st.session_state.detections)} affected areas")
                        for i, detection in enumerate(st.session_state.detections):
                            st.write(f"Detection #{i+1}: {detection['label']} ({detection['confidence']:.2f})")
                    else:
                        st.info("No specific disease regions detected by YOLO")
                
                if analysis_method == "Combined Analysis" or analysis_method == "Hybrid CNN Only":
                    # Show CNN results
                    st.subheader("Hybrid CNN Classification Results")
                    st.success(f"🌱 Classification: *{st.session_state.hybrid_label}*")
                    st.info(f"Confidence: {st.session_state.hybrid_conf:.2f}")
                    
                    # Show top 3 predictions
                    top_indices = np.argsort(st.session_state.predictions[0])[-3:][::-1]
                    st.write("Top 3 predictions:")
                    for idx in top_indices:
                        st.write(f"- {class_names[idx]}: {st.session_state.predictions[0][idx]:.2f}")
                    
                # Disease information and recommendations
                st.subheader("Treatment Recommendations")
                disease_name = st.session_state.hybrid_label

                
                # Extract plant and condition from the label
                parts = disease_name.split('_')
                if len(parts) >= 2:
                    plant = parts[0].replace('_', ' ')
                    condition = parts[1].replace('_', ' ')

                    
                    if "healthy" in condition.lower():
                        st.write(f"✅ Good news! Your {plant} appears to be healthy.")
                        st.write("Recommendations:")
                        st.write("- Continue with regular watering and fertilization")
                        st.write("- Monitor for any changes in leaf appearance")
                        st.write("- Maintain good air circulation around plants")
                    else:
                      if st.session_state.hybrid_conf >= 0.50:  # Only display disease if confidence is high
                          st.write(f"❗ Your {plant} appears to have: {condition}")
                          st.write("General recommendations:")
                          st.write("- Remove and destroy affected leaves")
                          st.write("- Improve air circulation around plants")
                          st.write("- Avoid overhead watering to reduce humidity")
                          st.write("- Consider appropriate fungicide/bactericide based on the specific disease")
                          st.write("- Consult with a local agricultural extension for specific treatment options")
    
    # About Page
    elif app_mode == "About":
        st.header("🌿 About This System")
        st.markdown(
            """
            ## Advanced Plant Disease Detection System
            
            This application combines two powerful AI technologies:
            
            ### 1. YOLO (You Only Look Once) Object Detection
            - Precisely locates and identifies diseased areas on plant leaves
            - Trained on annotated images to recognize specific disease patterns
            - Provides bounding boxes around affected areas with confidence scores
            
            ### 2. Hybrid CNN with Attention Mechanism
            - Deep learning model specifically designed for plant disease classification
            - Uses attention mechanism to focus on the most important features
            - Trained on the PlantVillage dataset with 38 different plant disease classes
            
            ### Technical Details
            - The YOLO model is built using the Ultralytics YOLOv8 framework
            - The Hybrid CNN uses TensorFlow with custom attention blocks
            - Both models work together to provide comprehensive disease analysis
            
            ### 📊 Dataset
            The system is trained on the PlantVillage dataset, which contains over 70,295 images of healthy and diseased plant leaves across various crops.
            
            ### 🔧 Development
            Developed using:
            - TensorFlow and Keras for deep learning
            - Ultralytics YOLO for object detection
            - OpenCV for image processing
            - Streamlit for the web interface
            
            For more information or to report issues, please contact the development team.
            """
        )

if __name__ == "__main__":
    main()