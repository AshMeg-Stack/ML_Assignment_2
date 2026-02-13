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
from xgboost import XGBClassifier


def main():
    df = pd.read_csv("data/heart.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/confusion_matrices", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

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
        ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42))
    ]

    rows = []

    for name, model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        y_prob = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
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

        cm = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred)

        pd.DataFrame(cm).to_csv(
            f"artifacts/confusion_matrices/confusion_matrix_{name}.csv",
            index=False
        )

        with open(f"artifacts/reports/classification_report_{name}.txt", "w") as f:
            f.write(rep)

        joblib.dump({"scaler": scaler, "model": model},
                    f"artifacts/models/{name}.pkl")

    pd.DataFrame(rows).to_csv("artifacts/metrics_table.csv", index=False)


if __name__ == "__main__":
    main()
