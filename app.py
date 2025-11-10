import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image

# --- App Title & Description ---
st.title("🍃 Guava Disease Classification")
st.write("Upload a guava fruit or leaf image to classify its disease.")

# --- Function to load the latest model from 'results/' ---
@st.cache_resource
def get_latest_model(results_dir="results"):
    model_files = [f for f in os.listdir(results_dir) if f.endswith(".keras")]
    if not model_files:
        raise FileNotFoundError("No .keras model files found in results/")
    latest = max(model_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    model_path = os.path.join(results_dir, latest)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# --- Load model once (cached) ---
model = get_latest_model()

# --- Define class names ---
class_names = [
    "anthracnose_fruit",
    "healthy_fruit",
    "fruitfly_fruit",
    "styler_end_root",
    "anthracnose_leaf",
    "healthy_leaf",
    "insect_bite_leaf",
    "multiple_leaf",
    "scorch_leaf",
    "yld_leaf"
]

# --- File Upload ---
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Display uploaded image ---
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Classifying... Please wait...")

    # --- Preprocess image ---
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Prediction ---
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # --- Display result ---
    st.success(f"✅ Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
