import json, joblib, numpy as np, pandas as pd, streamlit as st, shap, matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

@st.cache_resource
def load():
    model = joblib.load("models/xgboost.joblib")
    scaler = joblib.load("models/scaler.joblib")
    thr = json.load(open("models/thresholds.json"))["XGBoost"]
    return model, scaler, thr, shap.TreeExplainer(model)

model, scaler, thr, explainer = load()
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def prepare(df):
    X = df[FEATURES].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    return X

st.title("💳 Credit Card Fraud Detector")
st.caption("XGBoost model trained on the ULB / Worldline Credit Card Fraud dataset. Prototype for academic use.")
st.sidebar.header("Settings")
thr_use = st.sidebar.slider("Fraud threshold", 0.01, 0.99, float(min(max(thr, 0.01), 0.99)), 0.01,
                            help="Lower = catch more fraud but more false alarms.")
src = st.sidebar.radio("Input", ["Use sample transactions", "Upload CSV"])
if src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with columns Time, V1..V28, Amount", type="csv")
    data = pd.read_csv(up) if up else None
else:
    data = pd.read_csv("sample_transactions.csv")
    st.sidebar.info("50 transactions from the held-out test set (15 fraud, 35 legitimate).")

if data is None:
    st.info("Upload a CSV to start.")
    st.stop()
missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error(f"Missing columns: {missing[:5]}...")
    st.stop()

X = prepare(data)
data = data.copy()
data["fraud_probability"] = model.predict_proba(X)[:, 1]
data["prediction"] = np.where(data["fraud_probability"] >= thr_use, "FRAUD", "Legitimate")

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", len(data))
c2.metric("Flagged as fraud", int((data.prediction == "FRAUD").sum()))
if "Class" in data.columns:
    caught = int(((data.prediction == "FRAUD") & (data.Class == 1)).sum())
    c3.metric("Real fraud caught", f"{caught} / {int(data.Class.sum())}")

cols = ["Amount", "fraud_probability", "prediction"] + (["Class"] if "Class" in data.columns else [])
view = data[cols].sort_values("fraud_probability", ascending=False)
st.dataframe(view.style.format({"fraud_probability": "{:.3f}", "Amount": "{:.2f}"}), width="stretch", height=280)

st.subheader("Why was a transaction flagged?")
i = st.selectbox("Pick a transaction (row number)", list(view.index))
sv = explainer.shap_values(X.loc[[i]])[0]
top = pd.Series(sv, index=FEATURES).reindex(pd.Series(sv, index=FEATURES).abs().sort_values(ascending=False).index).head(8)
p = data.loc[i, "fraud_probability"]
st.write(f"**Fraud probability: {p:.1%}** → {data.loc[i, 'prediction']}")
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(top.index[::-1], top.values[::-1], color=["#c0392b" if v > 0 else "#2980b9" for v in top.values[::-1]])
ax.set_xlabel("SHAP value (red pushes towards fraud, blue towards legitimate)")
plt.tight_layout()
st.pyplot(fig)
st.caption("V1–V28 are anonymised PCA components, so their business meaning is unknown.")
