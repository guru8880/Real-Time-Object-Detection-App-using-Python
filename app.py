import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

import streamlit as st
import cv2
import sys

# Check and install missing packages
try:
    import numpy as np
except ImportError:
    st.error("NumPy not installed. Please run: pip install numpy==1.23.5")
    st.stop()

try:
    import torch
except ImportError:
    st.error("PyTorch not installed. Please run: pip install torch==2.0.1")
    st.stop()

try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics not installed. Please run: pip install ultralytics==8.0.186")
    st.stop()

# Title
st.title("YOLOv8 Real-Time Object Detection")

# Verify NumPy is working
try:
    np.array([1, 2, 3])  # Simple test
    st.sidebar.success("✓ NumPy is working")
except Exception as e:
    st.error(f"NumPy test failed: {e}")
    st.stop()

# Device selection
device_opt = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Using device: {device_opt}")

@st.cache_resource
def _load_model(name, device):
    model_map = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt", 
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt"
    }
    weights = model_map.get(name, "yolov8n.pt")
    
    try:
        model = YOLO(weights)
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar for model choice
model_size = st.sidebar.selectbox(
    "Select YOLOv8 Model Size",
    ("yolov8n", "yolov8s", "yolov8m", "yolov8l")
)

# Confidence threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Load model
model = _load_model(model_size, device_opt)

if model is None:
    st.error("Failed to load model. Please check the console for errors.")
    st.stop()

st.sidebar.success("Model loaded successfully!")

# Start webcam
if st.button("Start Detection"):
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            st.stop() 
        
        stframe = st.empty()
        stop_button = st.button("Stop Detection")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from camera.")
                break

            # Run YOLO inference with error handling
            try:
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame[:, :, ::-1], channels="RGB", use_column_width=True)
            except Exception as e:
                st.error(f"Inference error: {e}")
                break

            if stop_button:
                break

        cap.release()
        st.success("Camera feed stopped.")
        
    except Exception as e:
        st.error(f"Camera error: {e}")