
# ============================================================
# BANK MARKETING - MODEL DEVELOPMENT
# ============================================================



# ============================================================
#  IMPORT LIBRARIES
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
#  CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)

print("Model directory ready.")


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv(
    "bank-additional-full.csv",
    sep=";"
)

print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["y"]).copy()
y = data["y"].copy()


# ============================================================
# ENCODE TARGET VARIABLE
# ============================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

print("\nTarget classes:")
print(target_encoder.classes_)

# no  -> 0
# yes -> 1


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

label_encoders = {}

for column in X.columns:

    if X[column].dtype == "object":

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            X[column].astype(str)
        )

        label_encoders[column] = encoder


print("\nCategorical features encoded:")
print(list(label_encoders.keys()))


# ============================================================
#  TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split completed.")
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# DEFINE MODELS
# ============================================================

model = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------
    "Logistic Regression":
        LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ),

    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------
    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    # --------------------------------------------------------
    # k-Nearest Neighbors
    # --------------------------------------------------------
    "K-Nearest Neighbors":
        KNeighborsClassifier(),

    # --------------------------------------------------------
    # Naive Bayes
    # --------------------------------------------------------
    "Naive Bayes":
        GaussianNB(),

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
#  TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_model = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


for name, model in model.items():

    print("\nTraining:", name)

    # Train model
    model.fit(
        X_train_scaled,
        y_train
    )

    # Store trained model
    trained_model[name] = model

    # Generate predictions
    y_pred = model.predict(
        X_test_scaled
    )

    # Generate probability/decision scores for AUC
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        y_prob = model.decision_function(
            X_test_scaled
        )

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n", name)
    print("-" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))


# ============================================================
#  MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
)

comparison = comparison.round(4)

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# CONFUSION MATRIX AND CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRICES AND CLASSIFICATION REPORTS")
print("=" * 70)


for name, model in trained_model.items():

    y_pred = model.predict(
        X_test_scaled
    )

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_encoder.classes_,
            zero_division=0
        )
    )


# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

best_model_name = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best model based on Accuracy:",
    best_model_name
)


# ============================================================
# SAVE TRAINED MODELS
# ============================================================

joblib.dump(
    trained_model["Logistic Regression"],
    "model/logistic_regression.pkl"
)

joblib.dump(
    trained_model["Decision Tree"],
    "model/decision_tree.pkl"
)

joblib.dump(
    trained_model["K-Nearest Neighbors"],
    "model/knn.pkl"
)

joblib.dump(
    trained_model["Naive Bayes"],
    "model/naive_bayes.pkl"
)

joblib.dump(
    trained_model["Random Forest"],
    "model/random_forest.pkl"
)

print("\nAll ML models saved successfully.")


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("Scaler saved successfully.")


# ============================================================
#  SAVE FEATURE LABEL ENCODERS
# ============================================================

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

print("Feature label encoders saved successfully.")


# ============================================================
#  SAVE TARGET ENCODER
# ============================================================

joblib.dump(
    target_encoder,
    "model/target_encoder.pkl"
)

print("Target encoder saved successfully.")


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("Model comparison table saved successfully.")


# ============================================================
# SAVE ORIGINAL TEST DATA
# ============================================================

# X_test contains encoded/scaled data.
# We use the original dataset index to retrieve the
# original human-readable test rows.

test_data_original = data.loc[
    X_test.index
].copy()

test_data_original.to_csv(
    "test_data.csv",
    index=False
)

print("Original test data saved successfully.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated model files:")

print("  - logistic_regression.pkl")
print("  - decision_tree.pkl")
print("  - knn.pkl")
print("  - naive_bayes.pkl")
print("  - random_forest.pkl")
print("  - scaler.pkl")
print("  - label_encoders.pkl")
print("  - target_encoder.pkl")

print("\nGenerated project files:")

print("  - model_comparison.csv")
print("  - test_data.csv")

print("\nModel development pipeline completed.")
