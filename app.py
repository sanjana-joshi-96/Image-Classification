import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

from tensorflow.keras.models import load_model  # Correct import for load_model

# Load CIFAR-10 model
try:
    model = load_model('cifar10_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

@st.cache_resource
def load_mobilenetv2():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

@st.cache_resource
def load_cifar10_model():
    return tf.keras.models.load_model('cifar10_model.h5')

def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")#classifying
        model = load_mobilenetv2()
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for _, label, score in decoded_predictions:
            st.write(f"{label}: {score * 100:.2f}%")

def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        try:
            model = load_cifar10_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        img = image.resize((32, 32))
        img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10", "MobileNetV2 (ImageNet)"))
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
