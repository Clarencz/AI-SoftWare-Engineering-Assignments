# Part 2, Task 2: Deep Learning with TensorFlow
# Dataset: MNIST Handwritten Digits
# Goal: Build a CNN to classify digits with >95% accuracy and visualize results.

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def build_mnist_cnn():
    """
    Builds, trains, and evaluates a CNN model on the MNIST dataset.
    Saves the trained model to 'mnist_cnn.h5'.
    """
    print("--- Task 2: TensorFlow CNN on MNIST Dataset ---")

    # 1. Load and Preprocess Data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape data to add a "channel" dimension (1 for grayscale)
    # This is required for Conv2D layers.
    # (60000, 28, 28) -> (60000, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # 2. Build the CNN Model
    print("Building CNN model...")
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the 3D feature maps into a 1D vector
        layers.Flatten(),
        
        # Dense (fully connected) layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization
        
        # Output layer (10 classes for 10 digits)
        layers.Dense(10, activation='softmax')
    ])

    # 3. Compile the Model
    # We use 'sparse_categorical_crossentropy' because our labels (y_train)
    # are integers (0, 1, 2...) and not one-hot encoded.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 4. Train the Model
    print("\n--- Training Model ---")
    # We train for 5 epochs, which is typically enough for >98% accuracy on MNIST.
    # We use 10% of the training data for validation during training.
    model.fit(x_train, y_train, 
              epochs=5, 
              validation_split=0.1,
              batch_size=64)

    # 5. Evaluate the Model
    print("\n--- Evaluating Model on Test Data ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    
    if test_acc > 0.95:
        print("Achieved >95% test accuracy! 🎉")
    else:
        print("Did not achieve >95% accuracy. Try training for more epochs.")

    # 6. Save the Model
    # This is for the bonus task (Streamlit app)
    model_filename = 'mnist_cnn.h5'
    print(f"\nSaving trained model to {model_filename}...")
    model.save(model_filename)
    print("Model saved.")

    # 7. Visualize Predictions
    print("\n--- Visualizing Predictions ---")
    visualize_predictions(model, x_test, y_test)


def visualize_predictions(model, x_test, y_test, num_samples=5):
    """
    Plots sample images with their predicted and actual labels.
    """
    # Get predictions for the first 'num_samples' test images
    predictions = model.predict(x_test[:num_samples])
    
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        
        # Plot the image (remove the channel dimension for plotting)
        plt.imshow(np.squeeze(x_test[i]), cmap=plt.cm.binary)
        
        # Get the predicted class (the index with the highest probability)
        predicted_label = np.argmax(predictions[i])
        actual_label = y_test[i]
        
        # Set title color
        color = 'green' if predicted_label == actual_label else 'red'
        
        plt.title(f"Pred: {predicted_label}\nTrue: {actual_label}", color=color)
        plt.axis('off')

    print("Displaying prediction visualization. Close the plot window to exit.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    build_mnist_cnn()
