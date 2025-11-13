# ---
# Title: Part 2, Task 3: Predictive Analytics for Issue Priority
# Note: This is a .py file 
# ---

# ### Goal:
# Train a model to predict issue priority.
# As per the prompt, we will use the Kaggle Breast Cancer Dataset,
# and re-map its target (diagnosis) to represent "priority"
# (e.g., Malignant -> High Priority, Benign -> Low Priority).

# ### 1. Import Libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ### 2. Load and Preprocess Data

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create a DataFrame for easier manipulation
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Print dataset info
print("--- Dataset Head ---")
print(df.head())
print("\n--- Target Info ---")
print(f"Original target labels: {data.target_names}")
print(f"Original target distribution: \n{df['target'].value_counts()}")

# --- Re-mapping for the "Issue Priority" Task ---
# We will map the binary target to "High" and "Low" priority.
# 0 (malignant) -> 'High' (more severe, needs more attention)
# 1 (benign) -> 'Low' (less severe)

priority_map = {0: 'High', 1: 'Low'}
df['priority'] = df['target'].map(priority_map)

print("\n--- Re-mapped Target (Priority) ---")
print(df['priority'].value_counts())

# ### 3. Feature Engineering and Data Splitting

# Define features (X) and new target (y)
X = df[data.feature_names]
y = df['priority']

# Encode the new string labels
# 'High' -> 0, 'Low' -> 1
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded priority labels: {le.classes_}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale the features
# This is critical for many models, and good practice for all.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ### 4. Train a Random Forest Model

print("\n--- Training Model ---")
# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

print("Model training complete.")

# ### 5. Evaluate the Model

print("\n--- Model Evaluation ---")
# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Weighted F1-Score: {f1:.4f}")

# Display detailed classification report
# Note: 0 = 'High', 1 = 'Low'
target_names = le.classes_
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

