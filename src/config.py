"""
config.py
---------
Shared settings used by every script in this project.

Keeping paths and constants in one place means the other scripts stay short
and readable, and there is only ever one place to change something.
"""

from pathlib import Path

# ----- Project folders -------------------------------------------------------
# This file lives in src/, so the project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Make sure the output folders exist (safe to call repeatedly).
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# The single file where the trained model gets saved and later reloaded.
MODEL_PATH = MODELS_DIR / "model.joblib"

# ----- Reproducibility -------------------------------------------------------
# A fixed random seed means every run produces the exact same split and the
# exact same results. Without this, numbers would shift slightly each run and
# you could never tell whether a change you made actually mattered.
RANDOM_STATE = 42

# Fraction of data held back for the final, honest test.
TEST_SIZE = 0.20
