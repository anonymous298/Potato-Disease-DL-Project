import os
import streamlit as st
from src.pipe.prediction import predict  # Adjust the import based on your project structure

# Create temp directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

# Streamlit interface
st.title("Potato Disease Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    img_path = f"temp/{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make prediction
    predictions = predict(img_path)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Prediction:", predictions)