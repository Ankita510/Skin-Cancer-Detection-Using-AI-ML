import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:/Users/hp/Downloads/SkinCancer_Detection.keras")
    return model

model = load_model()

# UI part

st.title("🩺 Skin Cancer Detection (Malignant vs. Benign)")
st.write("Upload a skin lesion image, and the model will predict if it's **malignant** or **benign**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model
    img_resized = image.resize((150, 150))  # adjust if your model uses another input size
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred_prob = model.predict(img_array)[0][0]  # probability of malignant
    malignant_prob = float(pred_prob)
    benign_prob = 1 - malignant_prob

    # Results
    st.subheader("🔍 Prediction Results")
    st.write(f"**Malignant (Cancerous): {malignant_prob*100:.2f}%**")
    st.write(f"**Benign (Non-Cancerous): {benign_prob*100:.2f}%**")

    # Recommendation
    if malignant_prob >= 0.40:
        st.error("⚠️ Risk detected! Probability of malignant is above 40%. Please visit a hospital for further tests.")
    else:
        st.success("✅ Low risk. But keep monitoring and follow skin care precautions.")

    # Precautions Section
    st.subheader("🛡️ General Precautions")
    st.markdown("""
    - Avoid **excessive sun exposure**, use sunscreen (SPF 30+).
    - Monitor skin regularly for **new moles or changes**.
    - Maintain a **healthy diet** rich in fruits and vegetables.
    - Avoid tanning beds and **harsh chemicals** on the skin.
    - Schedule **regular dermatology checkups** if you are at higher risk.
    """)
