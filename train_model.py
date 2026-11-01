import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance


# ==========================================
# CONFIGURATION
# ==========================================

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# LOAD DATASET
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default="dataset/train.csv",
    help="Path to Kaggle training CSV"
)
args = parser.parse_args()

print("\nLoading dataset...")

df = pd.read_csv(args.data)
df.columns = df.columns.str.strip()

if "Loan_Status" not in df.columns:
    raise ValueError(
        "Loan_Status column not found. "
        "Please use the Kaggle training dataset."
    )

print("Dataset loaded:", df.shape)


# ==========================================
# DATA CLEANING
# ==========================================

df = df.replace(r"^\s*$", np.nan, regex=True)

# Remove unnecessary ID column
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Remove rows without target
df = df.dropna(subset=["Loan_Status"])

# Normalize target values
target = (
    df["Loan_Status"]
    .astype(str)
    .str.strip()
    .str.upper()
)

target_mapping = {
    "Y": 1,
    "N": 0,
    "YES": 1,
    "NO": 0,
    "1": 1,
    "0": 0
}

y = target.map(target_mapping)

if y.isna().any():
    raise ValueError("Unexpected values found in Loan_Status.")

y = y.astype(int)

X = df.drop(columns=["Loan_Status"])

# Remove columns with no useful information
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

if constant_columns:
    X = X.drop(columns=constant_columns)

if X.empty:
    raise ValueError("No usable input features found.")


# ==========================================
# IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Convert remaining numeric-looking values where appropriate
for col in categorical_features.copy():
    converted = pd.to_numeric(X[col], errors="coerce")

    if converted.notna().sum() == X[col].notna().sum():
        X[col] = converted
        numeric_features.append(col)
        categorical_features.remove(col)

# Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_transformer, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_transformer, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# DECISION TREE MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

print("\nTraining Decision Tree model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

labels = [0, 1]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=["Rejected", "Approved"],
    zero_division=0
))


# ==========================================
# PERMUTATION FEATURE IMPORTANCE
# ==========================================

print("\nCalculating feature importance...")

importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="accuracy",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance_result.importances_mean,
    "Std": importance_result.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance.csv"),
    index=False
)


# ==========================================
# SAVE INPUT SCHEMA
# ==========================================

schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "features": X.columns.tolist(),
    "defaults": {},
    "categories": {}
}

for col in numeric_features:
    median = X_train[col].median()
    schema["defaults"][col] = (
        float(median) if pd.notna(median) else 0.0
    )

for col in categorical_features:
    mode = X_train[col].mode(dropna=True)

    schema["defaults"][col] = (
        str(mode.iloc[0]) if not mode.empty else "Unknown"
    )

    values = (
        X_train[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    schema["categories"][col] = values

with open(
    os.path.join(ARTIFACT_DIR, "schema.json"),
    "w"
) as f:
    json.dump(schema, f, indent=4)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    os.path.join(ARTIFACT_DIR, "loan_model.pkl")
)


# ==========================================
# SAVE EVALUATION METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "labels": ["Rejected", "Approved"],
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "total_samples": int(len(df)),
    "feature_count": int(X.shape[1])
}

with open(
    os.path.join(ARTIFACT_DIR, "metrics.json"),
    "w"
) as f:
    json.dump(metrics, f, indent=4)


print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")
print("Saved files:")
print("- loan_model.pkl")
print("- metrics.json")
print("- schema.json")
print("- feature_importance.csv")
print("\nYou can now run the Streamlit application.")