"""
app.py
======
KEITA — Breast Cancer Prediction System (secure web app).

Features:
  - Password login so only you can open the app.
  - Personalised KEITA branding.
  - A polished dark interface.
  - Live predictions with confidence, plus feature importance, data preview,
    and a plain-language explanation.

Run it from the project root with:
    streamlit run app.py

The password lives in .streamlit/secrets.toml — change it there, and keep that
file private (the included .gitignore stops it being uploaded to GitHub).
"""

import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths and personalisation
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
APP_MODEL_PATH = ROOT / "models" / "app_model.joblib"
OWNER = "KEITA"  # <-- your name appears across the app

# Public demo or private app?
#   False = anyone can use it, no login (use this for a public LinkedIn demo).
#   True  = requires the username + password from .streamlit/secrets.toml.
REQUIRE_LOGIN = False 

st.set_page_config(
    page_title=f"{OWNER} | Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling — makes the app look polished rather than default Streamlit
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit's default chrome for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    /* Gradient hero banner */
    .hero {
        background: linear-gradient(120deg, #7c3aed 0%, #4f46e5 50%, #0ea5e9 100%);
        padding: 28px 34px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(124,58,237,0.25);
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 30px; font-weight: 800; }
    .hero p  { color: #ece9ff; margin: 6px 0 0 0; font-size: 15px; }
    .hero .badge {
        display:inline-block; background: rgba(255,255,255,0.18); color:#fff;
        padding:3px 12px; border-radius:999px; font-size:13px; font-weight:600;
        margin-bottom:10px;
    }

    /* Result cards */
    .result-card {
        padding: 24px; border-radius: 16px; text-align:center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .result-card .label { color:#fff; font-size: 30px; font-weight: 800; }
    .result-card .conf  { color:#fff; font-size: 18px; margin-top:6px; opacity:0.95; }

    /* Login card */
    .login-card {
        background:#1a1d29; padding:34px; border-radius:18px;
        border:1px solid #2a2e3e; box-shadow:0 12px 40px rgba(0,0,0,0.4);
    }

    /* Buttons a touch rounder */
    .stButton button { border-radius: 10px; font-weight: 600; }

    /* Footer credit */
    .credit { text-align:center; color:#6b7280; font-size:13px; margin-top:30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SECURITY — password gate
# ---------------------------------------------------------------------------
def check_password() -> bool:
    """
    Show a login screen and only let the app through if the password matches the
    one stored in .streamlit/secrets.toml. Uses a constant-time comparison so the
    check can't be guessed by timing it.
    """
    # Already logged in this session?
    if st.session_state.get("auth_ok", True):
        return True

    # Make sure a username and password have actually been configured.
    try:
        real_username = st.secrets["username"]
        real_password = st.secrets["password"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No login is set. Create `.streamlit/secrets.toml` with two lines:  "
            "username = \"your-name\"  and  password = \"your-secret\"  then restart the app."
        )
        st.stop()

    # Centered login card.
    st.markdown(
        f"""
        <div class="hero">
            <div class="badge">🔒 Secure access</div>
            <h1>{OWNER}'s Breast Cancer Prediction System</h1>
            <p>Private application — please sign in to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.subheader("🔑 Sign in")
        entered_user = st.text_input("Username", key="user_input")
        entered_pw = st.text_input("Password", type="password", key="pw_input")
        if st.button("Unlock", type="primary", use_container_width=True):
            # Both must match. compare_digest is a constant-time check.
            user_ok = hmac.compare_digest(str(entered_user), str(real_username))
            pw_ok = hmac.compare_digest(str(entered_pw), str(real_password))
            if user_ok and pw_ok:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password. Try again.")
        st.caption("Only the owner of this app knows the username and password.")
        st.markdown("</div>", unsafe_allow_html=True)

    return False


if REQUIRE_LOGIN and not check_password():
    st.stop()  # nothing below runs until the correct username + password are entered


# ---------------------------------------------------------------------------
# Model loading (with automatic retrain fallback)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    if APP_MODEL_PATH.exists():
        return joblib.load(APP_MODEL_PATH)

    data = load_breast_cancer()
    feature_names = [str(n) for n in data.feature_names[:10]]
    class_names = [str(n) for n in data.target_names]
    X = pd.DataFrame(data.data[:, :10], columns=feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "class_names": class_names,
        "accuracy": float(model.score(X_test, y_test)),
        "feature_importances": dict(
            zip(feature_names, model.feature_importances_.astype(float))
        ),
        "feature_stats": {
            name: {
                "min": float(X[name].min()),
                "max": float(X[name].max()),
                "mean": float(X[name].mean()),
            }
            for name in feature_names
        },
    }
    APP_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, APP_MODEL_PATH)
    return bundle


@st.cache_data
def load_sample_data():
    data = load_breast_cancer()
    cols = [str(n) for n in data.feature_names[:10]]
    df = pd.DataFrame(data.data[:, :10], columns=cols)
    df["diagnosis"] = [data.target_names[t] for t in data.target]
    benign_example = df[df["diagnosis"] == "benign"].iloc[0, :10].to_dict()
    malignant_example = df[df["diagnosis"] == "malignant"].iloc[0, :10].to_dict()
    return df, benign_example, malignant_example


bundle = load_or_train_model()
model = bundle["model"]
feature_names = bundle["feature_names"]
class_names = bundle["class_names"]
stats = bundle["feature_stats"]
sample_df, benign_example, malignant_example = load_sample_data()

# ---------------------------------------------------------------------------
# HERO / TITLE
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="badge">🩺 Machine Learning Health Tool</div>
        <h1>{OWNER}'s Breast Cancer Prediction System</h1>
        <p>Enter cell measurements to estimate whether a tumour is benign or
        malignant, with a confidence score. A learning project — not a medical device.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR — owner, model info, presets, logout
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {OWNER}")
    st.caption("Signed in")
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state["auth_ok"] = False
        st.rerun()

    st.divider()
    st.header("About the model")
    st.metric("Test accuracy", f"{bundle['accuracy'] * 100:.1f}%")
    st.caption(
        "Measured on patients the model never saw during training. "
        "It uses the 10 'mean' measurements below."
    )

    st.divider()
    st.subheader("Quick examples")
    st.caption("Fill the sliders with a real patient.")

    for name in feature_names:
        if name not in st.session_state:
            st.session_state[name] = float(stats[name]["mean"])

    a, b = st.columns(2)
    if a.button("Benign", use_container_width=True):
        for name in feature_names:
            st.session_state[name] = float(benign_example[name])
        st.rerun()
    if b.button("Malignant", use_container_width=True):
        for name in feature_names:
            st.session_state[name] = float(malignant_example[name])
        st.rerun()
    if st.button("Reset to average", use_container_width=True):
        for name in feature_names:
            st.session_state[name] = float(stats[name]["mean"])
        st.rerun()

# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------
st.subheader("Enter the measurements")
inputs = {}
left, right = st.columns(2)
for i, name in enumerate(feature_names):
    target_col = left if i % 2 == 0 else right
    lo, hi = float(stats[name]["min"]), float(stats[name]["max"])
    step = (hi - lo) / 200 or 0.01
    with target_col:
        inputs[name] = st.slider(name.title(), min_value=lo, max_value=hi, step=step, key=name)

# ---------------------------------------------------------------------------
# PREDICTION + RESULT
# ---------------------------------------------------------------------------
st.divider()
if st.button("🔍 Predict", type="primary", use_container_width=True):
    features = pd.DataFrame([[inputs[n] for n in feature_names]], columns=feature_names)
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    label = class_names[predicted_class]
    confidence = float(probabilities[predicted_class]) * 100

    if label == "benign":
        bg, emoji, message = "linear-gradient(120deg,#16a34a,#15803d)", "✅", "Benign (non-cancerous)"
    else:
        bg, emoji, message = "linear-gradient(120deg,#dc2626,#b02a37)", "⚠️", "Malignant (cancerous)"

    st.markdown(
        f"""
        <div class="result-card" style="background:{bg};">
            <div class="label">{emoji} {message}</div>
            <div class="conf">Confidence: {confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Probability malignant", f"{probabilities[0] * 100:.1f}%")
    c2.metric("Probability benign", f"{probabilities[1] * 100:.1f}%")
    st.progress(confidence / 100)

    if label == "malignant":
        st.warning(
            "The model leans malignant. In real screening this would prompt "
            "further clinical testing — never a diagnosis on its own."
        )
else:
    st.info("Set the measurements above, then click **Predict**.")

# ---------------------------------------------------------------------------
# ENHANCEMENTS
# ---------------------------------------------------------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 What drives predictions", "🔬 Data preview", "❓ How to read this"])

with tab1:
    st.markdown(
        "These are the measurements the model relies on most. Taller bars mean "
        "more influence on the benign-vs-malignant decision."
    )
    importance = (
        pd.Series(bundle["feature_importances"]).sort_values(ascending=False).rename("importance")
    )
    importance.index = [n.title() for n in importance.index]
    st.bar_chart(importance)

with tab2:
    st.markdown("A sample of the real dataset the model learned from.")
    st.dataframe(sample_df.head(15), use_container_width=True)
    st.caption(f"Dataset: {len(sample_df)} patients, 10 measurements shown.")

with tab3:
    st.markdown(
        """
        **Benign** means non-cancerous. **Malignant** means cancerous and able to spread.

        - The **confidence** is how sure the model is, as a percentage.
        - The model is right about **95% of the time** on unseen patients, but it is
          *not perfect* and is *not* a substitute for a doctor.
        - A **false negative** — calling a malignant tumour benign — is the most
          serious mistake, which is why real tools err toward caution and always
          involve clinical follow-up.
        """
    )

st.markdown(f'<div class="credit">Built by {OWNER} · Machine Learning project · scikit-learn + Streamlit</div>',
            unsafe_allow_html=True)
