import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('dog_vs_cat_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to 256x256
    image = np.array(image)  # Convert to array
    if image.shape[2] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    image = image / 255.0  # Normalize to [0, 1]
    image = image.reshape((1, 256, 256, 3))  # Reshape for model input
    return image

# Prediction function
def classify_image(image, model):
    test_input = preprocess_image(image)
    prediction = model.predict(test_input)[0][0]
    if prediction < 0.5:
        return "Dog 🐶"
    else:
        return "Cat 🐱"

# Streamlit app
st.title("Dog vs Cat Image Classifier")

st.write("Upload an image of a dog or cat to classify.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        result = classify_image(image, model)
        st.write(f"The model predicts: **{result}**")