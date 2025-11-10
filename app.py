import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Title ---
st.title("🍃 Guava Disease Classification")

st.write("Upload a guava fruit or leaf image to classify its disease.")

# --- Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # --- Preprocess image ---
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Load model ---
    model = tf.keras.models.load_model("final_model.keras")

    # --- Predict ---
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

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # --- Display result ---
    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")

