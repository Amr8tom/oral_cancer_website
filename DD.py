from PIL import Image
import numpy as np
from tensorflow import keras
from keras import models
import streamlit as st

def preprossing(image):
    Image = Image.resize(image, (50, 50))
    Image = Image.astype("float32") / 255.
    Image = Image.reshape((1,) + Image.shape)
    return Image

st.title("Oral Cancer Detection")
image_file = st.file_uploader("image upload", type=["png", "jpg", "jpeg"])

# Load model without compiling it
my_model = models.load_model("model87.h5", compile=False)

# Manually compile the model with the correct loss and reduction parameters
my_model.compile(optimizer="adam", 
                 loss=keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'), 
                 metrics=["accuracy"])

input_shape = my_model.layers[0].input_dtype[1:]
print('Input shape---------------------------------------:', input_shape)

def load_image(imageFile):
    img = Image.open(imageFile)
    return img

if image_file is not None:
    classes = ["non-cancer", "cancer"]
    st.image(load_image(image_file), width=256)
    
    # Load image file
    img = Image.open(image_file)
    
    # Resize image to 50x50 pixels
    img_resized = img.resize((50, 50))
    
    # Convert image to numpy array and normalize pixel values
    img_array = np.asarray(img_resized) / 255.0
    
    # Add an extra dimension to the image
    img_input = np.expand_dims(img_array, axis=0)

    # Print the shape of the input image
    print('Input shape:', img_input.shape)
    
    # Predict using the model
    result = my_model.predict(img_input)
    ind = np.argmax(result)
    final_output_prediction = classes[ind]
    
    print(final_output_prediction)
    st.header(final_output_prediction)
