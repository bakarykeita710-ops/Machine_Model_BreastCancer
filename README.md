# Breast Cancer Classifier — A Machine Learning Project

A complete, from-scratch machine learning project that predicts whether a
breast tumour is **malignant** or **benign** from 30 cell measurements. It is
built to be *understood*, not just run: every step is commented and explained,
and the three scripts walk through the full lifecycle of a real ML model.

**Final result:** ~98% accuracy on patients the model never saw during training.

---

## The one idea behind everything

Almost every supervised machine learning project — no matter the model — follows
the same spine:

```
   look at the data  →  split it  →  preprocess  →  train  →  evaluate honestly  →  use it
```

The model sitting in the middle is almost interchangeable. The discipline around
it is what makes the difference between a real result and a misleading one. This
project is organised into three "levels" that map onto that spine.

---

## What's in the project

```
breast_cancer_classifier/
├── README.md                     ← this file
├── requirements.txt              ← the libraries you need
├── app.py                        ← the Streamlit WEB APP (run this last)
├── src/
│   ├── config.py                 ← shared settings (paths, random seed)
│   ├── 01_explore_data.py        ← LEVEL 1: understand the data
│   ├── 02_train_model.py         ← LEVEL 2: train, evaluate, save the model
│   ├── 03_predict.py             ← LEVEL 3: use the model on new patients
│   └── train_app_model.py        ← trains the model the web app uses
├── models/                       ← the saved trained models land here
└── outputs/                      ← charts produced by the scripts land here
```

---

## Setup (one time)

From inside the project folder:

```bash
pip install -r requirements.txt
```

That installs scikit-learn (the ML library), pandas and numpy (for handling the
data), matplotlib (for the charts), and joblib (for saving the model).

Then run the three levels **in order**:

```bash
python src/01_explore_data.py
python src/02_train_model.py
python src/03_predict.py
```

---

## LEVEL 1 — Understand the data

**File:** `src/01_explore_data.py`
**What it does:** Loads the dataset and helps you *see* it before any modelling —
the equivalent of running descriptive statistics before an analysis.

**What to do:** Just run it and read the output.

**What to expect:**
- 569 patients, each with 30 numeric measurements (radius, texture, smoothness…).
- Two classes: **malignant (212 patients, 37%)** and **benign (357, 63%)**.
- A printed peek at the raw numbers and their summary statistics.
- A saved chart, `outputs/01_feature_distributions.png`, showing how a few key
  measurements differ between the two diagnoses. Visible separation between the
  two coloured histograms is a good sign — it means the measurements carry real
  signal the model can learn from.

**Why this level matters:** You never train on data you haven't looked at. The
class balance check here decides which evaluation metric you trust later.

---

## LEVEL 2 — Train, evaluate, and save the model

**File:** `src/02_train_model.py`
**What it does:** The heart of the project. It runs the entire training spine and
saves a finished model to disk.

**What to do:** Run it and read each section of the output top to bottom.

**What to expect, step by step:**

1. **Split** — 455 patients go to training, 114 are hidden away for testing.
   The model is only ever judged on those 114 unseen patients, because the goal
   is to work on *new* people, not to memorise known ones.

2. **Pipeline** — the script bundles two steps together: standardising every
   measurement to a z-score (subtract the mean, divide by the standard
   deviation — the same standardisation from statistics) and then logistic
   regression. Bundling them prevents *data leakage*, where information from the
   test set sneaks into training and makes results look better than they are.

3. **Cross-validation** — before trusting one test score, the training data is
   split into 5 parts and the model is trained and checked five times. Expect
   five similar scores averaging around **0.98**. Five *similar* scores means the
   model is reliable, not lucky.

4. **Training** — a single `.fit()` call. That one line is the actual "machine
   learning"; everything around it is preparation and judgement.

5. **Honest evaluation** — expect about **98% accuracy on the unseen test set**,
   plus:
   - An **overfitting check**: training accuracy and test accuracy are compared.
     A small gap (here ~0.01) means the model genuinely learned. A big gap would
     mean it memorised.
   - A **confusion matrix** breaking results into four outcomes. The dangerous
     error is a **false negative** — a malignant tumour called benign. On the
     test set the model makes only about one of these.
   - **Precision and recall** per class. In medical screening you care most
     about *recall* on malignant cases (catching every real cancer), because a
     missed cancer is far worse than a false alarm. Which metric matters is a
     judgement *you* make about the real-world cost of each mistake — the model
     cannot make it for you.

6. **A second model** — a Random Forest is trained in three lines to show that
   swapping models is trivial (every scikit-learn model shares the same
   `.fit()`/`.predict()` interface). On this dataset logistic regression actually
   edges it out, which is a useful reminder that fancier is not always better.

7. **Interpretability** — logistic regression reports a weight per measurement,
   so you can read off which measurements pushed toward malignant vs benign.
   A chart of the ten most influential ones is saved to
   `outputs/02_feature_importance.png`. This readability is a real advantage many
   models don't give you.

8. **Saving** — the whole pipeline (scaler + model together) is saved to
   `models/model.joblib`, along with the feature names and class names, so the
   next script can use it without retraining.

**Why this level matters:** This is where the honest-evaluation discipline lives.
The overfitting check and the confusion matrix are what separate someone who
understands ML from someone who just calls `.fit()`.

---

## LEVEL 3 — Use the model on new information

**File:** `src/03_predict.py`
**What it does:** Loads the saved model (no retraining) and feeds it new patients
to get a diagnosis plus a confidence level. This is what "putting a model to use"
actually looks like.

**What to do:** Run it as-is first to see the demonstration. Then open the file
and edit the numbers in `predict_custom_patient()` to test your own hypothetical
patient, and run it again.

**What to expect:**
- A demonstration where five real patients are fed to the model. For each, you
  see the model's diagnosis, its confidence, the full probability split, and —
  because we happen to know the truth for these — whether it was right. Expect
  all five correct, with confidence ranging from very high to merely confident.
- A "custom patient" prediction. Out of the box the custom values are set to the
  dataset's averages, which produces a deliberately uncertain result near 50/50 —
  an "average" tumour is genuinely ambiguous. Change a few values toward the
  malignant-leaning measurements (larger worst radius, worst area, worst concave
  points) and watch the probability swing.

**Why this level matters:** A model that can't take in new data is just an
exercise. This level closes the loop from raw measurements to a usable decision.

---

## LEVEL 4 — The interactive web app

**Files:** `src/train_app_model.py` (trains the app's model) and `app.py` (the app).
**What it does:** Turns the project into a web page where anyone can type in
measurements with sliders and get a live prediction with a confidence score.

**Why a separate model:** The app asks for only the 10 "mean" measurements so the
interface stays clean. To keep it honest, the app's model is trained on exactly
those 10 measurements — every slider genuinely affects the result, with nothing
faked in behind the scenes. It scores about **95% accuracy** on unseen patients.

**What to do:**

```bash
python src/train_app_model.py     # builds models/app_model.joblib (run once)
streamlit run app.py              # launches the web app in your browser
```

If you skip the first command, the app trains a model on its own the first time
it runs, so it always works.

**What to expect:** A browser tab opens with:
- a title and short description,
- ten sliders for the measurements, laid out in two columns,
- **Benign / Malignant / Reset** example buttons in the sidebar to fill the
  sliders with a real patient instantly,
- a **Predict** button that shows the diagnosis in a green (benign) or red
  (malignant) banner with a confidence percentage and full probability split,
- three tabs: a feature-importance chart, a preview of the real data, and a
  plain-language explanation for non-technical viewers.

To stop the app, press `Ctrl + C` in the terminal.

---

## Security & personalisation (the web app)

The app is branded to **KEITA** and protected by a password login, so only you
can open it.

**Your username and password live in `.streamlit/secrets.toml`:**

```toml
username = "keita"
password = "keita-change-this-2026"
```

Change both values to your own private details before using the app. The login screen now asks for a username AND a password, and both must match. The login
screen checks what's typed against this value using a constant-time comparison,
and nothing in the app loads until the correct password is entered.

**Keeping it private:** the included `.gitignore` already stops
`.streamlit/secrets.toml` from being uploaded to GitHub. Never paste your
password into `app.py` itself or share the secrets file.

**Honest limits of this protection:** a single shared password is good, simple
protection for a personal or student project. It is not the same as proper
multi-user accounts, and it is only as safe as keeping the password and the
secrets file private. If you ever deploy the app publicly, anyone with the link
will see the login screen but cannot get in without the password — so use a long,
unique password there.

**To rebrand** to a different name later, change the single `OWNER = "KEITA"`
line near the top of `app.py`.

---

## How to test it with your own data

Two ways, both in `src/03_predict.py`:

1. **Edit the custom patient.** The `custom` dictionary lists all 30
   measurements with the dataset's average as a starting value. Change any of
   them, rerun, and observe how the diagnosis and confidence move.

2. **Call the function directly.** `predict_patient(features, bundle)` takes a
   list of 30 numbers (in the order given by `bundle["feature_names"]`) and
   returns the label, the confidence, and the full probability split. You can
   import it into your own script and feed it any new patient.

---

## What the headline numbers mean

| Number | Meaning |
|--------|---------|
| ~0.98 test accuracy | 98% of unseen patients classified correctly |
| ~0.01 train-test gap | the model generalised; it did not just memorise |
| ~0.98 cross-val average | the result is stable, not a lucky split |
| 1 false negative | one malignant tumour was called benign on the test set |

---

## Where to take it next

- Swap in **gradient boosting** (`from sklearn.ensemble import
  GradientBoostingClassifier`) — the same three lines as the Random Forest.
- Adjust the **decision threshold** to chase higher recall on malignant cases,
  trading a few more false alarms for fewer missed cancers.
- Try the project on a different built-in dataset (e.g. `load_wine`) — the same
  spine works with almost no changes.

---

*Built with Python and scikit-learn. The model is trained on the Wisconsin
Diagnostic Breast Cancer dataset included with scikit-learn. This project is a
learning exercise and is not a medical device.*
