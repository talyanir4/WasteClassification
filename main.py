import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# constants
IMG_SIZE = (224, 224)
IMG_ADDRESS = "https://static.vecteezy.com/system/resources/previews/004/341/571/non_2x/waste-management-eco-friendly-living-2d-web-banner-poster-garbage-separation-man-and-woman-sorting-trash-flat-characters-on-cartoon-background-printable-patches-colorful-web-elements-vector.jpg"
IMAGE_NAME = "user_image.png"
CLASS_LABEL = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_LABEL.sort()

@st.cache_resource
def get_ConvNeXtXLarge_model():
    base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input, outputs=x)
    return model_frozen

@st.cache_resource
def load_sklearn_models(model_path):
    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)
    return final_model

def featurization(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)
    return predictions

# Load models
ConvNeXtXLarge_featurized_model = get_ConvNeXtXLarge_model()
classification_model = load_sklearn_models("MLP_best_model.pkl")

# Web app UI
st.title("Waste Classification")
st.image(IMG_ADDRESS, caption="Waste Classification")

st.subheader("Upload or Capture an Image")

# File uploader
image = st.file_uploader("Upload a Waste Image", type=["jpg", "png", "jpeg"])

# Camera input
camera_image = st.camera_input("Or Take a Photo")

# Process image from either source
if image or camera_image:
    user_image = Image.open(image if image else camera_image)

    # Save image to use in featurization
    user_image.save(IMAGE_NAME)

    # Display the user image
    st.image(user_image, caption="Selected Image")

    # Extract features and classify
    with st.spinner("Processing..."):
        image_features = featurization(IMAGE_NAME, ConvNeXtXLarge_featurized_model)
        model_predict = classification_model.predict(image_features)
        result_label = CLASS_LABEL[model_predict[0]]
        st.success(f"Prediction: {result_label}")
