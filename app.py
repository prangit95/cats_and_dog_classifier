import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image


# Load the pre-trained model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Steeamlit ui

st.title("Cat and Dog Image Classifier")
st.write("Upload an image of a cat or a dog to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    result = "Dog" if predictions[0][0] > 0.5 else "Cat"

st.subheader(f"Prediction:")
st.success(f"the model predicts {result}")

