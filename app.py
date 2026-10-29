import html
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

st.set_page_config(
    page_title="LeafLife | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1100px;}

.hero {
    background: linear-gradient(135deg, #14532d 0%, #16a34a 55%, #84cc16 100%);
    border-radius: 22px;
    padding: 2.4rem 2.6rem;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(22, 101, 52, 0.28);
    margin-bottom: 1.6rem;
}
.hero h1 {margin: 0; font-size: 2.8rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;}
.hero p {margin: 0.6rem 0 0 0; font-size: 1.1rem; color: rgba(255,255,255,0.92); max-width: 640px;}
.hero .pill {
    display: inline-block; margin: 1rem 0.4rem 0 0; padding: 0.25rem 0.85rem;
    border-radius: 999px; background: rgba(255,255,255,0.18);
    font-size: 0.85rem; font-weight: 600; color: #ffffff;
}

.card {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    height: 100%;
}
.card h4 {margin: 0 0 0.35rem 0; font-size: 1.05rem;}
.card p {margin: 0; opacity: 0.8; font-size: 0.95rem;}
.card .icon {font-size: 1.9rem; margin-bottom: 0.4rem;}

.result {
    border-radius: 18px;
    padding: 1.4rem 1.7rem;
    margin: 1.2rem 0 1.4rem 0;
    border: 1px solid rgba(128,128,128,0.22);
    border-left-width: 8px;
    background: rgba(128,128,128,0.07);
}
.result.ok {border-left-color: #16a34a;}
.result.bad {border-left-color: #ea580c;}
.result .label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.65;}
.result .name {font-size: 2rem; font-weight: 800; margin: 0.15rem 0 0.5rem 0;}
.badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
    font-size: 0.85rem; font-weight: 700; margin-right: 0.5rem;
}
.badge.ok {background: #dcfce7; color: #166534;}
.badge.bad {background: #ffedd5; color: #9a3412;}
.badge.plant {background: #e0f2fe; color: #075985;}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #16a34a;
    border-radius: 18px;
    background: rgba(22, 163, 74, 0.06);
}
[data-testid="stImage"] img {border-radius: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.15);}
[data-testid="stImageCaption"] {text-align: center; font-weight: 600;}

.footer-note {text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 2.5rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    """Load the model once and share it across all visitors."""
    model, transform, classes, device = load_trained_model(MODEL_PATH)
    # Grad-CAM registers hooks on the shared model, so only one prediction at a time
    return model, transform, classes, device, threading.Lock()


def clean(text: str) -> str:
    return text.replace("_", " ").strip()


def split_label(name: str):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight'); falls back gracefully."""
    if "___" in name:
        plant, condition = name.split("___", 1)
        return clean(plant), clean(condition)
    return "", clean(name)


model, transform, classes, device, lock = get_model()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🌿 LeafLife")
    st.caption("AI-powered plant disease detection")
    st.markdown("---")
    st.markdown("### 📸 Tips for best results")
    st.markdown(
        "- Use a **single leaf** in the frame\n"
        "- Good, even **daylight**\n"
        "- Keep the leaf **in focus**\n"
        "- Plain background if possible"
    )
    st.markdown("---")
    st.markdown("### 🧠 About the model")
    st.markdown(
        f"- Custom **CNN** built in PyTorch\n"
        f"- Recognises **{len(classes)} classes**\n"
        f"- Explained with **Grad-CAM**\n"
        f"- Running on **{str(device).upper()}**"
    )

# ---------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
  <h1>🌿 LeafLife</h1>
  <p>Upload a photo of a plant leaf and let deep learning spot the disease,
     then see exactly which parts of the leaf the model focused on.</p>
  <span class="pill">PyTorch CNN</span>
  <span class="pill">Grad-CAM explainability</span>
  <span class="pill">Instant results</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- upload
uploaded = st.file_uploader(
    "Drop a leaf image here (JPG or PNG, max 5 MB)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    steps = [
        ("📤", "1. Upload", "Add a clear photo of one leaf from your plant."),
        ("🔬", "2. Analyse", "The CNN classifies the leaf in a couple of seconds."),
        ("🔥", "3. Understand", "A Grad-CAM heatmap highlights the areas that drove the result."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), steps):
        with col:
            st.markdown(
                f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="footer-note">For educational purposes. Not a substitute for professional '
        "agricultural advice.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------- validate
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

# ---------------------------------------------------------------- predict
with st.spinner("Analysing leaf..."):
    try:
        with lock:
            class_name, overlay = predict_image(model, pil_image, transform, classes, device)
    except Exception:
        st.error("Prediction failed. Please try another image.")
        st.stop()

plant, condition = split_label(class_name)
healthy = "healthy" in class_name.lower()
state = "ok" if healthy else "bad"
status_text = "✅ Healthy" if healthy else "⚠️ Disease detected"
plant_badge = f'<span class="badge plant">🌱 {html.escape(plant)}</span>' if plant else ""

st.markdown(
    f"""
<div class="result {state}">
  <div class="label">Prediction</div>
  <div class="name">{html.escape(condition)}</div>
  <span class="badge {state}">{status_text}</span>{plant_badge}
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.image(pil_image, caption="Your leaf")
with col2:
    st.image(overlay, caption="Grad-CAM heatmap (warmer = more influence)")

st.markdown(
    '<div class="footer-note">For educational purposes. Not a substitute for professional '
    "agricultural advice.</div>",
    unsafe_allow_html=True,
)