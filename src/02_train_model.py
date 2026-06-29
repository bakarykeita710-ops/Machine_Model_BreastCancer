"""
02_train_model.py
================
LEVEL 2 — Train, evaluate honestly, and save the model.

This is the heart of the project. It follows the same spine every supervised
machine-learning task uses:

    load  ->  split  ->  build pipeline  ->  train  ->  evaluate  ->  save

Run with:
    python src/02_train_model.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from config import MODEL_PATH, OUTPUTS_DIR, RANDOM_STATE, TEST_SIZE


def main():
    # =====================================================================
    # STEP 1 — Load the data
    # =====================================================================
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")
    class_names = [str(n) for n in data.target_names]  # ['malignant', 'benign']

    # =====================================================================
    # STEP 2 — Split into training and test sets
    # ---------------------------------------------------------------------
    # We hide 20% of patients from the model during training. The model is
    # only ever judged on these unseen patients, because the real goal is to
    # work on NEW people, not to memorise the ones we already have answers for.
    # =====================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the malignant/benign ratio identical in both halves
    )
    print(f"Training patients: {len(X_train)}   |   Test patients: {len(X_test)}")

    # =====================================================================
    # STEP 3 — Build a pipeline (scaler + model bundled together)
    # ---------------------------------------------------------------------
    # A Pipeline chains steps so they always run in the right order. Here it
    # standardises every measurement (subtract mean, divide by std -- the same
    # z-score you know from statistics) and then feeds the result into logistic
    # regression. Bundling them this way makes data leakage almost impossible:
    # the scaler only ever learns from the training data inside each fold.
    # =====================================================================
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])

    # =====================================================================
    # STEP 4 — Cross-validate on the training set
    # ---------------------------------------------------------------------
    # Before trusting a single test score, we check stability: split the
    # TRAINING data into 5 parts, train on 4 and test on 1, five times over.
    # Five similar scores means the model is reliable, not lucky.
    # =====================================================================
    cv_scores = cross_val_score(logreg, X_train, y_train, cv=5, scoring="accuracy")
    print("\n5-fold cross-validation accuracy on training data:")
    print("  individual folds:", np.round(cv_scores, 3))
    print(f"  average: {cv_scores.mean():.3f}  (+/- {cv_scores.std():.3f})")

    # =====================================================================
    # STEP 5 — Train on the full training set
    # ---------------------------------------------------------------------
    # One line. Everything around it is preparation and judgement; THIS is the
    # actual "machine learning" -- finding the best weight for each measurement.
    # =====================================================================
    logreg.fit(X_train, y_train)

    # =====================================================================
    # STEP 6 — Evaluate honestly on the held-out test set
    # =====================================================================
    y_pred = logreg.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = logreg.score(X_train, y_train)

    print("\n" + "=" * 60)
    print("RESULTS — Logistic Regression")
    print("=" * 60)
    print(f"Accuracy on training data: {train_acc:.3f}")
    print(f"Accuracy on UNSEEN test data: {test_acc:.3f}")

    # Overfitting check: a big gap between the two means the model memorised
    # instead of learning. A small gap (like here) means it generalises well.
    gap = train_acc - test_acc
    verdict = "healthy (generalises well)" if gap < 0.05 else "warning: possible overfitting"
    print(f"Gap between them: {gap:.3f}  ->  {verdict}")

    # The confusion matrix breaks the result into the four possible outcomes.
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"                  pred malignant   pred benign")
    print(f"  actual malignant      {cm[0, 0]:>4}            {cm[0, 1]:>4}")
    print(f"  actual benign         {cm[1, 0]:>4}            {cm[1, 1]:>4}")
    print("\n  A false NEGATIVE (malignant called benign) is the dangerous error:")
    print(f"  the model made {cm[0, 1]} of those on the test set.")

    # Precision / recall / f1 per class.
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # =====================================================================
    # STEP 7 — Compare against a second model (Random Forest)
    # ---------------------------------------------------------------------
    # Because scikit-learn gives every model the same .fit()/.predict()
    # interface, swapping models is trivial. Trees do not need scaling.
    # =====================================================================
    forest = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    forest.fit(X_train, y_train)
    forest_acc = forest.score(X_test, y_test)
    print("=" * 60)
    print(f"Random Forest test accuracy: {forest_acc:.3f}")
    print(f"Logistic Regression test accuracy: {test_acc:.3f}")
    print("=" * 60)

    # =====================================================================
    # STEP 8 — Look at what the model learned (interpretability)
    # ---------------------------------------------------------------------
    # Logistic regression hands us a weight per measurement. Positive weights
    # push toward 'benign' (class 1), negative toward 'malignant' (class 0).
    # =====================================================================
    coefs = pd.Series(
        logreg.named_steps["model"].coef_[0], index=X.columns
    ).sort_values()

    print("\nTop 5 measurements pushing toward MALIGNANT:")
    print(coefs.head(5).round(3).to_string())
    print("\nTop 5 measurements pushing toward BENIGN:")
    print(coefs.tail(5).round(3).to_string())

    # Save a chart of the 10 most influential measurements.
    top = pd.concat([coefs.head(5), coefs.tail(5)])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Most influential measurements\n(red = pushes malignant, green = pushes benign)")
    ax.set_xlabel("model weight")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "02_feature_importance.png", dpi=120)
    print(f"\nSaved feature-importance chart to: {OUTPUTS_DIR / '02_feature_importance.png'}")

    # =====================================================================
    # STEP 9 — Save the trained pipeline for later use
    # ---------------------------------------------------------------------
    # We save the WHOLE pipeline (scaler + model together) so that when we
    # predict on new patients later, the exact same scaling is reapplied
    # automatically. We also store the feature names and class names so the
    # prediction script knows the correct column order.
    # =====================================================================
    bundle = {
        "pipeline": logreg,
        "feature_names": list(X.columns),
        "class_names": class_names,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("\nDone. Next: python src/03_predict.py")


if __name__ == "__main__":
    main()
