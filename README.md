# Implementation of ML Model for Image Classification

This project demonstrates the implementation of machine learning models for image classification using TensorFlow and Streamlit. It includes two models: MobileNetV2 (ImageNet) and CIFAR-10, allowing users to classify uploaded images through an interactive web interface.

## Features
- **MobileNetV2**: Pre-trained on ImageNet for general image classification.
- **CIFAR-10**: Custom-trained model for classifying CIFAR-10 dataset images.
- Interactive web app built using Streamlit for easy user interaction.

## Prerequisites
- Python 3.7–3.11
- TensorFlow
- Streamlit
- Pillow
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Implementation-of-ML-Model-for-Image-Classification
## Install the required dependencies
  ```bash
  pip install -r requirements.txt
````
## Usuage
- Run the Streamlit app:
```bash
   streamlit run app.py
```
- Upload an image in the web app.
- Select the model (MobileNetV2 or CIFAR-10) for classification.
- View the predicted class and confidence score.
