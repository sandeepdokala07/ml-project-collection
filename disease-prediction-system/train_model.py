"""
train_model.py
---------------
Trains a Random Forest classifier to predict disease from a binary symptom
vector, evaluates it, and saves the trained model + label encoder + feature
importances to disk for use by the Streamlit app / CLI.

Run:  python train_model.py
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/symptom_disease_dataset.csv"
MODEL_PATH = "models/disease_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    symptom_cols = [c for c in df.columns if c != "disease"]

    X = df[symptom_cols]
    y = df["disease"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y_encoded, cv=5)

    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"5-fold CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

    # Feature importances (which symptoms matter most overall)
    importances = pd.Series(model.feature_importances_, index=symptom_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 most predictive symptoms overall:")
    print(importances.head(10))

    # Persist model artifacts
    artifact = {
        "model": model,
        "encoder": encoder,
        "symptom_cols": symptom_cols,
        "feature_importances": importances,
        "test_accuracy": acc,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
