"""
Module 5 Week A — Integration: ML Evaluation Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                   "num_support_calls", "senior_citizen",
                   "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


# =========================
# Task 1
# =========================
def load_and_prepare(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # target
    y = df["churned"]

    
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    X = df.drop(columns=["churned"])

    return X, y


# =========================
# Task 2
# =========================
def build_preprocessor():
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor


# =========================
# Task 3
# =========================
def define_models():
    preprocessor = build_preprocessor()

    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="saga",
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(
                alpha=1.0,
                class_weight="balanced"
            ))
        ]),

        "Dummy_most_frequent": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Dummy_stratified": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42))
        ]),
    }

    return models


# =========================
# Task 4
# =========================
def evaluate_models(models, X, y, cv=5, random_state=42):

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    scoring = ["accuracy", "precision", "recall", "f1"]

    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv_split,
            scoring=scoring
        )

        results.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
        })

    return pd.DataFrame(results)


# =========================
# Task 5
# =========================
def final_evaluation(pipeline, X_train, X_test, y_train, y_test):

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


# =========================
# Recommendation
# =========================
def recommend_model(results_df):

    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))

   
    real_models = results_df[~results_df["model"].str.contains("Dummy")]

    best_model_name = real_models.sort_values(by="f1_mean", ascending=False).iloc[0]["model"]

    print("\n=== Best Model ===")
    print(best_model_name)

    print("\n=== Recommendation ===")
    print(
        f"The recommended model is {best_model_name} because it achieved the highest F1 score. "
        f"Accuracy alone is not reliable in this problem because the Most-frequent Dummy model "
        f"can achieve high accuracy while failing to detect churners. The selected model provides "
        f"a better balance between precision and recall, which is important since missing a "
        f"churning customer is more costly than a false positive. Additionally, the model clearly "
        f"outperforms the stratified dummy baseline, indicating it has learned meaningful patterns "
        f"beyond random guessing."
    )

    return best_model_name


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data = load_and_prepare()

    if data is not None:
        X, y = data

        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = define_models()

        results = evaluate_models(models, X_train, y_train)

        best_model_name = recommend_model(results)

        # Task 5
        best_pipeline = models[best_model_name]

        test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)

        print("\n=== Final Test Metrics ===")
        print(test_metrics)