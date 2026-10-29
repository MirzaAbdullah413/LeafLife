import io
import os
import threading

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from Predict_Image import load_trained_model, predict_image

torch.set_num_threads(1)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaf_disease_model.pth")
MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_PIXELS = 25_000_000           # blocks "decompression bomb" images

st.set_page_config(page_title="LeafLife", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Load the model once and share it across all visitors."""
    model, transform, classes, device = load_trained_model(MODEL_PATH)
    # Grad-CAM registers hooks on the shared model, so only one prediction at a time
    return model, transform, classes, device, threading.Lock()


def pretty(name: str) -> str:
    return name.replace("___", " - ").replace("_", " ")


st.title("🌿 LeafLife")
st.write("Upload a photo of a plant leaf to detect its disease. "
         "The heatmap shows which regions influenced the prediction (Grad-CAM).")

model, transform, classes, device, lock = get_model()

uploaded = st.file_uploader("Choose a leaf image (JPG or PNG, max 5 MB)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    if uploaded.size > MAX_FILE_SIZE:
        st.error("Image too large (max 5 MB).")
        st.stop()

    try:
        img = Image.open(io.BytesIO(uploaded.getvalue()))
        if img.width * img.height > MAX_PIXELS:
            st.error("Image dimensions too large.")
            st.stop()
        pil_image = img.convert("RGB")
        pil_image.thumbnail((640, 640))
    except (UnidentifiedImageError, OSError):
        st.error("Invalid or corrupted image. Please try another one.")
        st.stop()

    with st.spinner("Analysing leaf..."):
        try:
            with lock:
                class_name, overlay = predict_image(model, pil_image, transform, classes, device)
        except Exception:
            st.error("Prediction failed. Please try another image.")
            st.stop()

    st.success(f"Prediction: **{pretty(class_name)}**")

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Your image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)
