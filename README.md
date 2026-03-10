Dog vs Cat Classifier (Transfer Learning)

Introduction
This project is a simple Artificial Intelligence application that classifies images of dogs and cats using Deep Learning and Transfer Learning techniques.
The system is built using Python and TensorFlow/Keras. A pre-trained convolutional neural network is used as the base model to perform image classification. Instead of training a neural network from scratch, transfer learning allows the model to reuse knowledge from large pretrained datasets and adapt it for the dog vs cat classification task.
The project also includes a graphical user interface (GUI) built with Tkinter. Users can select an image from their computer and the system will predict whether the image contains a dog or a cat along with the confidence score.
This project demonstrates how machine learning models can be integrated into simple desktop applications.

Features
Image classification using Transfer Learning
Dog vs Cat prediction
Simple graphical user interface (GUI)
Model loading and real-time prediction
Built with TensorFlow/Keras

Project Structure
AI_TransferLearning
│
├── train.py          # Train the deep learning model
├── predict.py        # Run prediction from image
├── test_tf.py        # TensorFlow test script
├── app_gui.py        # GUI application for image classification
├── model.h5          # Trained model
├── .gitignore
└── README.md
Technologies Used
Python
TensorFlow
Keras
Transfer Learning
Tkinter
NumPy
Pillow

Dataset
The dataset used for training contains images of dogs and cats.
Due to file size limitations, the dataset is not included in this repository.
You can download a similar dataset here:
https://www.kaggle.com/datasets/tongpython/cat-and-dog

Installation Guide
1. Clone the repository
git clone https://github.com/StylingOnZ/AI_TransferLearning.git
cd AI_TransferLearning
2. Install required libraries
Install dependencies using pip:
pip install tensorflow numpy pillow opencv-python
Training the Model
Run the training script:
python train.py
This will train the model and save the trained model as:
model.h5
Run Prediction
To run prediction using a trained model:
python predict.py
Run the GUI Application
Launch the graphical interface:
python app_gui.py
A window will appear where you can upload an image and the system will classify whether the image is a dog or a cat.
