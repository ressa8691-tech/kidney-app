import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
MODEL_PATH = "kidney_model.h5"

st.title("🧠 Kidney Analysis System")

# =========================
# LOAD MODEL SAFELY
# =========================
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Put kidney_model.h5 in this folder.")
    st.stop()

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# =========================
# INPUT OPTIONS
# =========================
st.subheader("Choose Input Method")

option = st.radio("Select input:", ["Upload Image", "Use Camera"])

image = None

# =========================
# UPLOAD OPTION
# =========================
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# =========================
# CAMERA OPTION
# =========================
elif option == "Use Camera":
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# =========================
# PROCESS IMAGE
# =========================
if image is not None:
    
    st.image(image, caption="Input Image", use_column_width=True)

    try:
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(img)[0][0]

        st.subheader("📊 Result")

        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"⚠️ Abnormal Detected\nConfidence: {confidence:.2f}%")

            # Risk level
            if confidence > 80:
                st.error("🔴 High Risk")
            elif confidence > 60:
                st.warning("🟠 Medium Risk")
            else:
                st.info("🟡 Low Risk")

        else:
            confidence = (1 - prediction) * 100
            st.success(f"✅ Normal\nConfidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")