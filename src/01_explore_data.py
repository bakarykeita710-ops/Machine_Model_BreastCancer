"""
01_explore_data.py
==================
LEVEL 1 — Understand the data before touching any model.

Run this first. It does NOT train anything. Its only job is to load the data
and help you SEE what you are working with, exactly like running descriptive
statistics before any analysis.

Run with:
    python src/01_explore_data.py
"""

import matplotlib
matplotlib.use("Agg")  # render plots to a file instead of a screen (works on any machine)
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.datasets import load_breast_cancer

from config import OUTPUTS_DIR


def main():
    # ---- Load the dataset -------------------------------------------------
    # This dataset ships inside scikit-learn, so there is nothing to download.
    # It contains 569 real tumour samples, each described by 30 measurements.
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)  # the 30 inputs
    y = pd.Series(data.target, name="diagnosis")             # the known answer

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Patients (rows):        {X.shape[0]}")
    print(f"Measurements (columns): {X.shape[1]}")
    print(f"Target classes:         {[str(n) for n in data.target_names]}  (0=malignant, 1=benign)")

    # ---- Class balance ----------------------------------------------------
    # How many of each diagnosis do we have? If one class massively outnumbered
    # the other, plain accuracy would become misleading, so we always check.
    print("\nClass balance:")
    counts = y.value_counts().sort_index()
    for value, count in counts.items():
        name = data.target_names[value]
        pct = 100 * count / len(y)
        print(f"  {name:<10} {count:>3} patients  ({pct:.1f}%)")

    # ---- A peek at the actual numbers ------------------------------------
    print("\nFirst 5 patients, first 6 measurements:")
    print(X.iloc[:5, :6].round(2).to_string())

    print("\nSummary statistics (first 6 measurements):")
    print(X.iloc[:, :6].describe().round(2).to_string())

    # ---- Save a simple visual --------------------------------------------
    # One clear picture: how a few key measurements differ between the two
    # diagnoses. Separation here is a good sign the model can learn from them.
    key_features = ["mean radius", "mean texture", "mean concave points"]
    fig, axes = plt.subplots(1, len(key_features), figsize=(13, 4))

    for ax, feature in zip(axes, key_features):
        ax.hist(X[y == 0][feature], bins=25, alpha=0.6, label="malignant")
        ax.hist(X[y == 1][feature], bins=25, alpha=0.6, label="benign")
        ax.set_title(feature)
        ax.set_xlabel("value")
        ax.set_ylabel("number of patients")
        ax.legend()

    fig.suptitle("How key measurements differ by diagnosis")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / "01_feature_distributions.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved a plot to: {out_path}")
    print("\nDone. You now understand the data. Next: python src/02_train_model.py")


if __name__ == "__main__":
    main()
