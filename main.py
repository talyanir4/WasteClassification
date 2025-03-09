import streamlit as st
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from collections import defaultdict

# Constants
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

def extract_indexes(prob_list: list) -> tuple:
    process_list = prob_list.copy()
    process_list.sort(reverse=True)
    first_prob, second_prob = process_list[0], process_list[1]
    first_index = prob_list.index(first_prob)
    second_index = prob_list.index(second_prob)
    return ((first_prob, second_prob), (first_index, second_index))

# Load models
ConvNeXtXLarge_featurized_model = get_ConvNeXtXLarge_model()
classification_model = load_sklearn_models("MLP_best_model.pkl")

def contamination():
    CD1 = 0.2
    CD2 = 0.4
    contamined_dict = defaultdict(list)
    image_features = featurization(IMAGE_NAME, ConvNeXtXLarge_featurized_model)
    probs = classification_model.predict_proba(image_features)
    pred_probs, pred_index = extract_indexes(list(probs[0]))

    if (pred_probs[0] - pred_probs[1]) < CD1:
        contamined_dict["status"].append("Highly Contaminated")
    elif (pred_probs[0] - pred_probs[1]) >= CD1 and (pred_probs[0] - pred_probs[1]) < CD2:
        contamined_dict["status"].append("Low Contamination")
    else:
        contamined_dict["status"].append("Not Contaminated")

    contamined_dict["labels"].append(f"{CLASS_LABEL[pred_index[0]]} and {CLASS_LABEL[pred_index[1]]}")
    return contamined_dict

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Sort", "History"])

if page == "Sort":
    st.title("RecycleSense AI")
    st.image(IMG_ADDRESS, caption="Waste Classification")
    st.subheader("Upload or Capture an Image")

    image = st.file_uploader("Upload a Waste Image", type=["jpg", "png", "jpeg"])
    camera_image = st.camera_input("Or Take a Photo")

    current_time = time.time()

    if image or camera_image:
        user_image = Image.open(image if image else camera_image)
        user_image.save(IMAGE_NAME)
        st.image(user_image, caption="Selected Image")

        with st.spinner("Processing..."):
            results = contamination()
            st.success(f"Contamination level: {results['status'][0]}, Prediction labels: {results['labels'][0]}")
            pred_time = time.time()
            lag = pred_time - current_time
            st.success(f"Prediction Time: {lag}")
            # Save result to history
            st.session_state.history.append({
                "image": IMAGE_NAME,
                "status": results['status'][0],
                "labels": results['labels'][0],
                "image_data": user_image
            })

elif page == "History":
    st.title("History")
    if st.session_state.history:
        for index, record in enumerate(st.session_state.history):
            with st.expander(f"Prediction {index+1}"):
                st.image(record["image_data"], caption=f"Prediction {index+1}")
                st.write(f"**Contamination Level:** {record['status']}")
                st.write(f"**Prediction Labels:** {record['labels']}")
    else:
        st.write("No history available.")

elif page == "About":
    st.title("About")
    st.write("RecycleSense AI uses machine learning and visual recognition to sort waste and detect contamination, optimizing recycling accuracy and reducing greenhouse gas emissions. It deploys a multilayer perceptron (MLP) deep learning model for efficient waste sorting that can be implemented at materials recovery facilities.")
