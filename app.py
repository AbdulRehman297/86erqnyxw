import streamlit as st
from skin_model import predict_skin_disease
from groq_api import get_recommendation
import os

# Streamlit App Title
st.title("Skin Disease Detector")

# Upload image
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    
    image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    
    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    # Predict disease
    disease, confidence = predict_skin_disease(image_path)
    st.write(f"### Diagnosis: {disease}")
    #st.write(f"**Confidence Level:** {confidence}%")

    # recommendations
    with st.spinner("Fetching treatment recommendations..."):
        recommendation = get_recommendation(disease)

    
    st.subheader("Treatment Recommendations")
    st.write(recommendation)

    os.remove(image_path)
