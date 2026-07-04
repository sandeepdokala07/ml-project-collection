"""
predictor.py
-------------
Loads the trained model artifacts and exposes a clean predict() function
that turns a set of selected symptoms into a ranked list of likely diseases
with probabilities.
"""

import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "models/disease_model.pkl"

_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def get_symptom_list() -> list:
    """Return the full list of symptoms the model was trained on."""
    return _load_artifact()["symptom_cols"]


def get_model_metrics() -> dict:
    """Return stored evaluation metrics from training."""
    artifact = _load_artifact()
    return {
        "test_accuracy": artifact["test_accuracy"],
        "cv_accuracy_mean": artifact["cv_accuracy_mean"],
        "cv_accuracy_std": artifact["cv_accuracy_std"],
    }


def get_top_symptoms(n: int = 10):
    """Return the n most globally predictive symptoms (feature importance)."""
    artifact = _load_artifact()
    return artifact["feature_importances"].head(n)


def predict(selected_symptoms: list, top_k: int = 3) -> list:
    """
    Predict the most likely disease(s) given a list of selected symptom names.

    Returns a list of (disease_name, probability) tuples, sorted descending,
    length top_k.
    """
    artifact = _load_artifact()
    model = artifact["model"]
    encoder = artifact["encoder"]
    symptom_cols = artifact["symptom_cols"]

    vector = np.zeros(len(symptom_cols))
    selected_set = set(selected_symptoms)
    for i, symptom in enumerate(symptom_cols):
        if symptom in selected_set:
            vector[i] = 1

    vector_df = pd.DataFrame([vector], columns=symptom_cols)
    probs = model.predict_proba(vector_df)[0]
    disease_names = encoder.classes_

    ranked = sorted(zip(disease_names, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
