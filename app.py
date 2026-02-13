import os
import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Prediction - ML Assignment 2")

metrics_path = "artifacts/metrics_table.csv"
models_dir = "model/saved_models"
cm_dir = "artifacts/confusion_matrices"
report_dir = "artifacts/reports"
test_csv_path = "data/test_data.csv"

if os.path.exists(metrics_path):
    st.subheader("Model Metrics (Hold-out Test Split)")
    st.dataframe(pd.read_csv(metrics_path))

if os.path.exists(test_csv_path):
    with open(test_csv_path, "rb") as f:
        st.download_button("Download test_data.csv", f, file_name="test_data.csv")

if not os.path.exists(models_dir):
    st.error("Models not found. Run: python -m model.train_and_eval")
    st.stop()

model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
model_names = [f.replace(".pkl", "") for f in model_files]
model_name = st.selectbox("Select Model", model_names)

cm_path = f"{cm_dir}/confusion_matrix_{model_name}.csv"
if os.path.exists(cm_path):
    st.subheader("Confusion Matrix")
    st.dataframe(pd.read_csv(cm_path, header=None))

report_path = f"{report_dir}/classification_report_{model_name}.txt"
if os.path.exists(report_path):
    st.subheader("Classification Report")
    with open(report_path, "r") as f:
        st.text(f.read())

uploaded = st.file_uploader("Upload test CSV (features only)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)

    bundle = joblib.load(os.path.join(models_dir, model_name + ".pkl"))
    scaler = bundle["scaler"]
    model = bundle["model"]

    X = scaler.transform(df)
    preds = model.predict(X)

    out = df.copy()
    out["prediction"] = preds

    st.subheader("Predictions")
    st.dataframe(out)
