"""
03_predict.py
============
LEVEL 3 — Use the saved model on NEW information.

This script does NOT retrain anything. It loads the model you already trained
and saved, then feeds it new patients to get a diagnosis and a confidence
level. This is what "putting a model into use" actually looks like.

Run with:
    python src/03_predict.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer

from config import MODEL_PATH


def load_model():
    """Load the saved bundle (pipeline + feature names + class names)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No saved model found. Run 'python src/02_train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


def predict_patient(features, bundle):
    """
    Predict the diagnosis for ONE patient.

    `features` is a list/array of the 30 measurements in the correct order
    (the same order as bundle['feature_names']).

    Returns the predicted label and the probability the model assigns to it.
    """
    pipeline = bundle["pipeline"]
    class_names = bundle["class_names"]

    # The model expects a 2D table (rows x columns), so wrap the single
    # patient in another list -> shape (1, 30).
    X_new = pd.DataFrame([features], columns=bundle["feature_names"])

    predicted_class = pipeline.predict(X_new)[0]          # 0 or 1
    probabilities = pipeline.predict_proba(X_new)[0]      # e.g. [0.98, 0.02]

    label = class_names[predicted_class]
    confidence = probabilities[predicted_class]
    return label, confidence, probabilities


def demo_with_real_examples(bundle):
    """
    Borrow a few real patients from the dataset to act as 'new' arrivals.

    These specific patients are only used here for demonstration. The point is
    to show the model handling concrete cases and to compare its answer against
    the TRUE diagnosis so you can see it is actually right.
    """
    data = load_breast_cancer()
    class_names = bundle["class_names"]

    # Pick a few patients by index: a couple clearly malignant, a couple benign.
    example_indices = [0, 5, 19, 100, 568]

    print("=" * 64)
    print("DEMONSTRATION — feeding the model real patients it must diagnose")
    print("=" * 64)

    for idx in example_indices:
        features = data.data[idx]
        true_label = class_names[data.target[idx]]

        label, confidence, probs = predict_patient(features, bundle)
        correct = "CORRECT" if label == true_label else "WRONG"

        print(f"\nPatient #{idx}")
        print(f"  model says : {label}  ({confidence * 100:.1f}% confident)")
        print(f"  truth      : {true_label}   ->  {correct}")
        print(f"  full probabilities: "
              f"malignant {probs[0] * 100:.1f}% / benign {probs[1] * 100:.1f}%")


def predict_custom_patient(bundle):
    """
    Test the model on numbers YOU choose.

    Edit the values in `custom` below to represent a hypothetical patient,
    then rerun the script. All 30 measurements are listed with the dataset's
    own average value as a starting point, so you can change a few and watch
    the prediction move.
    """
    # These defaults are roughly the dataset averages -- a 'neutral' patient.
    # Change any of them to see how the diagnosis and confidence shift.
    custom = {
        "mean radius": 14.13,          "mean texture": 19.29,
        "mean perimeter": 91.97,       "mean area": 654.89,
        "mean smoothness": 0.096,      "mean compactness": 0.104,
        "mean concavity": 0.089,       "mean concave points": 0.048,
        "mean symmetry": 0.181,        "mean fractal dimension": 0.063,
        "radius error": 0.405,         "texture error": 1.217,
        "perimeter error": 2.866,      "area error": 40.34,
        "smoothness error": 0.007,     "compactness error": 0.025,
        "concavity error": 0.032,      "concave points error": 0.012,
        "symmetry error": 0.021,       "fractal dimension error": 0.004,
        "worst radius": 16.27,         "worst texture": 25.68,
        "worst perimeter": 107.26,     "worst area": 880.58,
        "worst smoothness": 0.132,     "worst compactness": 0.254,
        "worst concavity": 0.272,      "worst concave points": 0.115,
        "worst symmetry": 0.290,       "worst fractal dimension": 0.084,
    }

    # Put the values in the exact order the model expects.
    features = [custom[name] for name in bundle["feature_names"]]
    label, confidence, probs = predict_patient(features, bundle)

    print("\n" + "=" * 64)
    print("YOUR CUSTOM PATIENT (edit the numbers in predict_custom_patient)")
    print("=" * 64)
    print(f"  diagnosis : {label}  ({confidence * 100:.1f}% confident)")
    print(f"  malignant {probs[0] * 100:.1f}% / benign {probs[1] * 100:.1f}%")


def main():
    bundle = load_model()
    demo_with_real_examples(bundle)
    predict_custom_patient(bundle)
    print("\nDone.")


if __name__ == "__main__":
    main()
