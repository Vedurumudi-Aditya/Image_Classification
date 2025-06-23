# Image Classification
# Classifies images as cat or dog using a convolutional neural network with TensorFlow

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

# Generate synthetic image dataset (replace with real dataset path)
def load_synthetic_data(num_samples=100, img_size=64):
    images = np.random.rand(num_samples, img_size, img_size, 3) * 255
    labels = np.random.randint(0, 2, num_samples)  # 0: cat, 1: dog
    return images.astype(np.uint8), labels

# Load data
images, labels = load_synthetic_data()

# Preprocess data
images = images / 255.0
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Example prediction
sample_image = X_test[0:1]
prediction = model.predict(sample_image)
print(f"Prediction: {'Dog' if prediction[0][0] > 0.5 else 'Cat'}")
