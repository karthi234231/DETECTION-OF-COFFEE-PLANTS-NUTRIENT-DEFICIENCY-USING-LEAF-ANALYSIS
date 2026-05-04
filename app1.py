# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO

# # Paths
# MODEL_PATH = "weights/best.pt"
# RESULT_FOLDER = "static/results"

# # Ensure result folder exists
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # Load YOLOv8 Model
# model = YOLO(MODEL_PATH)

# # Streamlit UI
# st.title("YOLOv8 Object Detection")
# st.write("Upload an image and detect objects using YOLOv8.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Convert to PIL image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Save image for processing
#     image_path = os.path.join(RESULT_FOLDER, uploaded_file.name)
#     image.save(image_path)

#     # Run YOLO Detection
#     results = model(image_path)
#     result_image = results[0].plot()  # Annotated image

#     # Convert OpenCV result to PIL
#     result_pil = Image.fromarray(result_image[..., ::-1])

#     # Save output
#     result_path = os.path.join(RESULT_FOLDER, f"result_{uploaded_file.name}")
#     result_pil.save(result_path)

#     # Display detected objects
#     st.image(result_pil, caption="Detected Objects", use_column_width=True)

#     st.success("Detection Completed!")

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64

# Paths
MODEL_PATH = "weights/best.pt"
RESULT_FOLDER = "static/results"
BG_IMAGE_PATH = "bg.jpg"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 Model
model = YOLO(MODEL_PATH)

# Function to encode image to Base64 for CSS
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Apply background image using CSS
def set_bg_image(bg_image_path):
    base64_str = get_base64_of_image(bg_image_path)
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        color: brown !important;
    }}
    .image-container {{
        display: flex;
        justify-content: center;
        gap: 30px;
    }}
    .header-text {{
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        color: brown !important;
    }}
    .stMarkdown, .stTextInput, .stButton, .stFileUploader {{
        color: brown !important;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Set background image
set_bg_image(BG_IMAGE_PATH)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: brown;'>YOLOv8 Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: brown;'>Upload an image and detect objects using YOLOv8.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to PIL image
    image = Image.open(uploaded_file)

    # Resize image to reduce size (e.g., max width/height of 600px)
    image.thumbnail((600, 600))

    # Save resized image for processing
    image_path = os.path.join(RESULT_FOLDER, uploaded_file.name)
    image.save(image_path)

    # Run YOLO Detection
    results = model(image_path)

    # Get annotated image
    result_image = results[0].plot()

    # Convert to PIL image
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)

    # Resize result image to match input size
    result_pil.thumbnail((600, 600))

    # Save output
    result_path = os.path.join(RESULT_FOLDER, f"result_{uploaded_file.name}")
    result_pil.save(result_path)

    # Layout for side-by-side images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='header-text'>Preview</p>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<p class='header-text'>Detection</p>", unsafe_allow_html=True)
        st.image(result_pil, caption="Detected Objects", use_column_width=True)

    st.markdown("<p style='color: brown; font-weight: bold;'>Detection Completed!</p>", unsafe_allow_html=True)