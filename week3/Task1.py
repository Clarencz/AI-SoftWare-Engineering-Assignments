# Part 2, Task 1: Classical ML with Scikit-learn
# Dataset: Iris Species Dataset
# Goal: Train a decision tree classifier and evaluate it.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def run_iris_classifier():
    """
    Loads the Iris dataset, preprocesses it, trains a Decision Tree,
    and evaluates its performance.
    """
    print("--- Task 1: Scikit-learn Decision Tree on Iris Dataset ---")

    # 1. Load Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")

    # 2. Preprocess Data
    # Note: The assignment mentions "handle missing values," but the Iris dataset
    # is clean and has no missing values.
    # Note: The assignment mentions "encode labels," but 'y' is already
    # encoded as 0, 1, 2.
    
    # We will scale the features, which is good practice (though not
    # strictly necessary for Decision Trees, it's vital for other models).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Split Data
    # Splitting the data into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data into {len(X_train)} training and {len(X_test)} test samples.")

    # 4. Train a Decision Tree Classifier
    print("Training Decision Tree Classifier...")
    # We use random_state for reproducibility
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate the Model
    print("\n--- Model Evaluation ---")
    y_pred = classifier.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Calculate Precision, Recall, and F1-score
    # We use 'weighted' average to account for any class imbalance (though Iris is balanced)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print(f"Weighted Precision: {precision:.2f}")
    print(f"Weighted Recall: {recall:.2f}")

    # Display detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

if __name__ == "__main__":
    run_iris_classifier()
