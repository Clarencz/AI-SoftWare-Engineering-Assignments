AI Tools Assignment: Mastering the AI Toolkit
This repository contains the complete submission for the "AI Tools and Applications" assignment. The project demonstrates proficiency in Scikit-learn, TensorFlow, and spaCy, along with an analysis of AI ethics and model deployment.

Project Structure

Document 4.pdf: Contains all answers to theoretical questions (Part 1) and the ethics/optimization analysis (Part 3).

task_1.py: (Part 2, Task 1) A Python script for classifying the Iris dataset using Scikit-learn.

task_2.py: (Part 2, Task 2) A Python script that builds and trains a CNN on the MNIST dataset using TensorFlow. This script also saves the trained model as mnist_cnn.h5.

task_3.py: (Part 2, Task 3) A Python script demonstrating NER and sentiment analysis on sample text using spaCy.

deploy.py: (Bonus Task) A Streamlit web application that loads the trained mnist_cnn.h5 model and allows users to draw a digit for classification.

requirements.txt: A list of necessary Python packages to run these scripts.

How to Run

1. Setup

First, clone the repository and install the required packages:

git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
# This will install: scikit-learn, tensorflow, spacy, spacytextblob,
# streamlit, streamlit-drawable-canvas, pandas (for canvas), opencv-python-headless
#
# You also need to download the spaCy model:
python -m spacy download en_core_web_sm
python -m textblob.download_corpora


2. Run Practical Tasks

Task 1: Scikit-learn (Iris)

python task_1.py


This will print the accuracy, precision, and recall of the Decision Tree model.

Task 2: TensorFlow (MNIST)

python task_2.py


This will train the CNN, print its test accuracy (should be >95%), and show a plot of 5 sample predictions. It will also create a file named mnist_cnn.h5.

Task 3: spaCy (NER & Sentiment)

python task_3.py


This will print the extracted entities (Products/Brands) and the sentiment for several sample reviews.

3. Run Bonus Web App

Ensure you have already run task_2_mnist_tensorflow.py to generate the mnist_cnn.h5 model file.

streamlit run deploy.py


This will open a new tab in your browser with the interactive digit classifier.

Report & Video

Report: The report.md file contains all written answers. This can be converted to a PDF for submission.

Video: A 3-minute video presentation explaining our approach and showcasing the practical tasks (especially the Streamlit app) will be created and shared on the Community platform.

Group Members
Clarence Mabeya