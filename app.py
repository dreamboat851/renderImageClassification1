import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model.savedmodel')

# Streamlit app title
st.write("""
         # Fruit Classifier
         """
         )
st.write("This is a simple image classification web app to classify fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Function to preprocess and predict
def import_and_predict(image_data, model):
    size = (224, 224)  # Resize target dimensions
    try:
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Resize with Pillow
    except AttributeError:  # Fallback for older Pillow versions
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)  # Convert to NumPy array
    normalized_image = image / 255.0  # Normalize pixel values to [0, 1]
    img_reshape = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    prediction = model.predict(img_reshape)  # Get predictions
    return prediction

# Handling file uploads and displaying results
if file is None:
    st.text("Please upload an image file.")
else:
    # Open and display the uploaded image
    uploaded_image = Image.open(file)
    st.image(uploaded_image, use_column_width=True)

    # Make prediction
    prediction = import_and_predict(uploaded_image, model)

    # Display the classification results
    if np.argmax(prediction) == 0:
        st.write("It is an apple!")
        st.write("Usually comes in red and green color.")
    elif np.argmax(prediction) == 1:
        st.write("It is a banana!")
    elif np.argmax(prediction) == 2:
        st.write("It is a kiwi")
    elif np.argmax(prediction) == 3:
        st.write("It is a lemon")
    elif np.argmax(prediction) == 4:
        st.write("It is a mango")
    elif np.argmax(prediction) == 5:
        st.write("It is a orange")
    elif np.argmax(prediction) == 6:
        st.write("It is a pear")
    elif np.argmax(prediction) == 7:
        st.write("It is a pineapple")
    elif np.argmax(prediction) == 8:
        st.write("It is a pomegranate")
    elif np.argmax(prediction) == 9:
        st.write("It is a watermelon")

    st.text("Probability (0: Apple, 1: Banana 2: Kiwi, 3: Lemon 4: Mango, 5: Orange 6: Pear, 7: Pineapple 8: Pomegranate, 9: Watermelon):")
    st.write(prediction)
