import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

model = MobileNetV2(weights="imagenet")
print("Model loaded successfully!")
