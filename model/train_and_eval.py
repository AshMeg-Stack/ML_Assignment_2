import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except:
    XGB_OK = False


def main():
    df = pd.read_csv("data/heart.csv")

    if "target" not in df.columns:
        raise ValueError("heart.csv must have a column named 'target'")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)

    X_test.to_csv("data/test_data.csv", index=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000)),
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=7)),
        ("NaiveBayes", GaussianNB()),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]

    if XGB_OK:
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss"
            )
        ))
    else:
        print("XGBoost not available. Install using: pip install xgboost")

    rows = []
    best_name = None
    best_f1 = -1
    best_bundle = None

    for name, model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = float("nan")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        rows.append({
            "Model": name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc
        })

        print("\n==============================")
        print("Model:", name)
        print("Accuracy:", round(acc, 4))
        print("AUC:", round(auc, 4) if not np.isnan(auc) else "nan")
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(f1, 4))
        print("MCC:", round(mcc, 4))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        joblib.dump({"scaler": scaler, "model": model}, f"artifacts/models/{name}.pkl")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_bundle = {"scaler": scaler, "model": model}

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("artifacts/metrics_table.csv", index=False)

    joblib.dump(best_bundle, "artifacts/best_model.pkl")

    print("\nSaved:")
    print("data/test_data.csv")
    print("artifacts/metrics_table.csv")
    print("artifacts/best_model.pkl")
    print("artifacts/models/*.pkl")
    print("Best model:", best_name, "F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
