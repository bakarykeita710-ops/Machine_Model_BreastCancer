"""
train_app_model.py
==================
Trains the model used by the Streamlit web app.

Why a separate model from 02_train_model.py?
The web app asks the user for the 10 "mean" measurements only, so the UI stays
clean. To keep the app honest, the model is trained on exactly those same 10
measurements -- every slider the user moves genuinely affects the prediction,
with no hidden values faked in behind the scenes.

It also saves the extra things the app needs to build a good interface:
  - test accuracy (to display)
  - feature importances (Random Forest gives these for free)
  - the realistic min / max / average of each measurement (to set slider ranges)

Run with:
    python src/train_app_model.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE

APP_MODEL_PATH = MODELS_DIR / "app_model.joblib"


def main():
    data = load_breast_cancer()

    # The first 10 columns are the "mean" measurements -- the ones the app asks for.
    feature_names = [str(n) for n in data.feature_names[:10]]
    class_names = [str(n) for n in data.target_names]  # ['malignant', 'benign']

    X = pd.DataFrame(data.data[:, :10], columns=feature_names)
    y = pd.Series(data.target, name="diagnosis")

    # Standard train/test split so we can report an honest accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Random Forest: strong on tabular data, needs no scaling, and reports
    # feature importance which the app displays.
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Honest evaluation on unseen patients.
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("Confusion matrix [[TM, FB], [FM, TB]]:\n", cm)

    # Per-feature statistics so the app's sliders have sensible ranges/defaults.
    feature_stats = {
        name: {
            "min": float(X[name].min()),
            "max": float(X[name].max()),
            "mean": float(X[name].mean()),
        }
        for name in feature_names
    }

    importances = dict(zip(feature_names, model.feature_importances_.astype(float)))

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "class_names": class_names,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
        "feature_stats": feature_stats,
    }
    joblib.dump(bundle, APP_MODEL_PATH)
    print(f"\nSaved app model to: {APP_MODEL_PATH}")
    print("Now run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
