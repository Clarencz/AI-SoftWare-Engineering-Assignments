# Part 2: Task 1 - Edge AI Prototype

## Project Goal
#Train a lightweight MobileNetV2 model to classify images (simulated as recyclable items), convert it to TensorFlow Lite for edge deployment, and demonstrate inference.

## Implementation Code


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

# 1. SETUP & DATA GENERATION (Simulated Dataset)
print("Generating dummy data for simulation...")
# Shape: (100 images, 224 height, 224 width, 3 channels)
X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
y_train = np.random.randint(0, 2, size=(100,)) # Binary: 0 or 1

# 2. DEFINE LIGHTWEIGHT MODEL (MobileNetV2)
# MobileNet is optimized for low-latency edge devices
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze base layers for transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. TRAIN MODEL
print("Training model...")
model.fit(X_train, y_train, epochs=1, verbose=1)

# 4. CONVERT TO TENSORFLOW LITE (Edge AI Step)
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization: Quantization (Reduces model size by ~4x)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('recycling_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Success! Model saved as 'recycling_model.tflite'.")
print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")

# 5. SIMULATE INFERENCE ON EDGE
interpreter = tf.lite.Interpreter(model_path="recycling_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with random input
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Inference Result (Probability): {output_data[0][0]:.4f}")