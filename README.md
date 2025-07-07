# CIFAR-10 Image Classification using CNN in TensorFlow

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### 📦 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

### 📊 Dataset
- 50,000 training images
- 10,000 test images
- 10 image classes

### 🧠 Model Architecture
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- 2 Dense layers (final output: 10 classes)

### 🏃‍♂️ Training
- 20 epochs
- Optimizer: Adam
- Loss Function: SparseCategoricalCrossentropy
- Final Training Accuracy: ~85%
- Final Validation Accuracy: ~71%

### 📈 Visualization
- Accuracy and Loss graphs are plotted for both training and validation data using Matplotlib.
- Sample test predictions are visualized using `imshow`.

### 🚀 How to Run
```python
%pip install tensorflow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
