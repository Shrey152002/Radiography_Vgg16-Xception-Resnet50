# Radiography_Vgg16-Xception-Resnet50
Multiclass Classification on Radiography Dataset
This project implements a multiclass classification model using three popular deep learning architectures—ResNet50, XceptionNet, and VGG16—to classify radiography images into categories: COVID, Lung Opacity, Normal, and Viral Pneumonia.

Table of Contents
Project Overview
Setup
Model Architectures
Training
Results
Project Overview
The project aims to classify radiography images into four categories:

COVID
Lung Opacity
Normal
Viral Pneumonia
The models used in this project are:

ResNet50
XceptionNet
VGG16
Setup
Requirements
Python 3.x
TensorFlow or Keras
NumPy
Matplotlib
OpenCV (optional for image preprocessing)
Install dependencies
bash
Copy code
pip install -r requirements.txt
Dataset
The dataset consists of images of chest X-rays labeled into four categories:

COVID
Lung Opacity
Normal
Viral Pneumonia
Ensure that the dataset is placed in the data/ directory, with separate folders for each class.

Model Architectures
ResNet50: A deep residual network to capture image features effectively.
XceptionNet: An efficient model with depthwise separable convolutions for improved performance.
VGG16: A simpler CNN architecture with deep layers designed for image recognition tasks.
Each model is fine-tuned with a pre-trained base on ImageNet and further trained on the radiography dataset.

Training
The training script trains all three models on the dataset. The following command can be used to run the training:

bash
Copy code
python train.py
Training parameters such as learning rate, epochs, and batch size can be modified in the script.

Results
The models' performance is evaluated using accuracy, precision, recall, and F1-score. You can view the results for each model after training completes.
