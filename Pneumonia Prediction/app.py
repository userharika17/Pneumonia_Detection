import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load the trained pneumonia detection model
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
model = load_model(MODEL_PATH, compile=False)

# Load the labels
class_names = [line.strip() for line in open(LABELS_PATH, "r").readlines()]

# Set Streamlit page config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)
    
    # Preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize the image
    normalized_image_array = (image_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display result
    st.write(f"**Prediction:** {class_name.strip()} (Confidence: {confidence_score:.2f})")
    if "PNEUMONIA" in class_name.upper():
        st.error(f"⚠️ **Pneumonia Detected!** Confidence: {confidence_score:.2f}")
    else:
        st.success(f"✅ **No Pneumonia Detected.** Stay healthy! Confidence: {confidence_score:.2f}")

