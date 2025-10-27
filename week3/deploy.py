# Bonus Task: Deploy Your Model with Streamlit
# This script creates a web interface for the MNIST classifier.
#
# To run:
# 1. Make sure 'mnist_cnn.h5' (from task_2) is in the same directory.
# 2. Run: streamlit run bonus_task_mnist_streamlit.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import cv2 # OpenCV for image processing

# --- Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🧠",
    layout="wide"
)

# --- Model Loading ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    """Loads the pre-trained MNIST CNN model."""
    try:
        model = tf.keras.models.load_model('mnist_cnn.h5')
        return model
    except (IOError, ImportError) as e:
        st.error(f"Error loading model 'mnist_cnn.h5'. Did you run task_2_mnist_tensorflow.py first?")
        return None

model = load_model()

# --- Page Title and Introduction ---
st.title("🖌️ MNIST Handwritten Digit Classifier")
st.markdown("Draw a single digit (0-9) in the box below and click 'Classify' to see the model's prediction.")
st.markdown("This model is a Convolutional Neural Network (CNN) built with TensorFlow/Keras.")

# --- Main Layout (2 columns) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Draw Your Digit Here")
    
    # --- Drawable Canvas ---
    # We create a 280x280 canvas (10x the MNIST 28x28 size)
    # This makes it easier for the user to draw.
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Black background (will be inverted)
        stroke_width=20,  # A good thickness for drawing digits
        stroke_color="#FFFFFF", # White stroke
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    
    # --- Classify Button ---
    if st.button("Classify", use_container_width=True):
        if canvas_result.image_data is not None and model is not None:
            # 1. Get image data from canvas
            # This is a 4-channel (RGBA) image
            img = canvas_result.image_data
            
            # 2. Convert to Grayscale
            # We take just the alpha channel (or could convert to grayscale)
            # but since stroke is white and bg is black, alpha works well.
            # We use cv2.cvtColor to get a single channel grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            # 3. Resize to 28x28
            # This is the size the model expects.
            # We use INTER_AREA interpolation for shrinking.
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # 4. Invert Colors (if necessary)
            # Our canvas is black bg, white stroke.
            # MNIST is white bg, black stroke.
            # The model was trained on (0=black, 255=white).
            # Our drawing is (0=black, 255=white). So no inversion needed.
            # If canvas was white bg, black stroke, we'd need:
            # img_resized = cv2.bitwise_not(img_resized)

            # 5. Normalize and Reshape
            # Normalize from [0, 255] to [0, 1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Reshape to (1, 28, 28, 1) for the model (1 sample)
            img_final = np.reshape(img_normalized, (1, 28, 28, 1))

            # 6. Make Prediction
            prediction = model.predict(img_final)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            st.success(f"Predicted Digit: {predicted_digit}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            
            # 7. Show Probability Chart
            st.subheader("Probability Distribution")
            # We convert the prediction (numpy array) to a simple list for the chart
            prob_df = {str(i): prob for i, prob in enumerate(prediction[0])}
            st.bar_chart(prob_df)

            # 8. Show the processed image
            st.subheader("Image Sent to Model (28x28)")
            st.image(img_normalized, caption="Processed 28x28 image", width=140)

        else:
            st.warning("Please draw a digit first.")
