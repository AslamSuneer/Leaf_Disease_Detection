# Leaf Disease Detection

This repository is dedicated to the development and deployment of a **Leaf Disease Detection System** that leverages multiple deep learning models for plant disease classification. The goal of this project is to provide a reliable tool to help farmers detect plant diseases at an early stage, allowing for timely intervention to improve crop production.

## Project Overview

The system integrates multiple models in the following order:

1. **CNN** (Basic): A Convolutional Neural Network for disease classification.
2. **Hybrid CNN**: A hybrid CNN model that enhances the basic CNN architecture with additional techniques, improving classification performance.
3. **YOLO**: A state-of-the-art object detection model for detecting leaves and diseases.
4. **YOLO-Hybrid**: A hybrid model combining YOLO and Hybrid CNN to benefit from both object detection and classification, providing the best accuracy and efficiency.

The project utilizes the **PlantVillage dataset** and provides a web-based interface using **Streamlit** to interact with the model.

## Features

- **Real-time disease detection**: Detect and classify leaf diseases with high accuracy.
- **Treatment recommendations**: Based on the detected disease, the system provides potential treatment options.
- **User-friendly interface**: Streamlit-based interface to easily upload images and view results.

## Installation

### Prerequisites

- Python 3.13 or higher
- Git
- Anaconda (for managing environments)
- Streamlit
- PyTorch
- TensorFlow
- OpenCV
- Git LFS (for large file storage)




Leaf_Disease_Detection/
│
├── app.py                          # Streamlit web app
├── models/                          # Folder containing trained models
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── data/                           # Folder for dataset (if applicable)
