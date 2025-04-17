import streamlit as st
import time
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("🌿 LEAF DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown(
        """
        ## Welcome to the Leaf Disease Recognition System! 🍃🔍
        
        Keeping your plants healthy is essential for a successful harvest. Our AI-powered system helps identify plant diseases quickly and accurately.
        
        ### 🌱 How It Works:
        1. **Upload an Image** - Go to the **Disease Recognition** page and upload a clear image of a plant leaf.
        2. **AI Analysis** - The system will analyze the image and detect possible diseases using cutting-edge machine learning techniques.
        3. **Instant Results** - Get a detailed diagnosis and suggestions to keep your plants healthy.
        
        ### ✅ Why Choose Our System?
        - **High Accuracy** - Powered by advanced deep learning technology.
        - **Fast & Efficient** - Get results in just seconds.
        - **Easy to Use** - Simple interface for everyone, from farmers to researchers.
        
        🔍 **Ready to begin?** Head over to the **Disease Recognition** page and start diagnosing your plants today!
        """
    )

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🌿 Disease Recognition")
    test_image = st.file_uploader("📷 Upload an Image:")
    
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    if test_image and st.button("Predict Disease"):
        st.write("🍂 **Leaves rustling... Analyzing the image!**")  # Nature-themed animation message
        
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        with st.spinner('🍃 Analyzing the image... Please wait!'):
            time.sleep(3)
        
        st.success(f"🌱 The model predicts: **{class_name[result_index]}**")