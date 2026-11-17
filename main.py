
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

print("=" * 60)
print("       IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n[1] Loading Dataset...")

df = pd.read_csv("iris_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Distribution:")
print(df["species"].value_counts())

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\n[2] Performing Exploratory Data Analysis...")

sns.pairplot(df, hue="species", diag_kind="hist")
plt.savefig("outputs/pairplot.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species")
plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.savefig("outputs/species_count.png", dpi=300)
plt.close()

print("EDA graphs saved successfully!")

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

print("\n[3] Data Preprocessing...")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# 5. MODEL TRAINING
# ==========================================

print("\n[4] Training Machine Learning Models...")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])
}

results = []

# ==========================================
# 6. MODEL EVALUATION
# ==========================================

print("\n[5] Model Evaluation")

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"outputs/{filename}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ==========================================
# 7. MODEL COMPARISON
# ==========================================

print("\n[6] Model Comparison")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv("outputs/model_comparison.csv", index=False)

# ==========================================
# 8. DECISION BOUNDARY VISUALIZATION
# ==========================================

print("\n[7] Creating Decision Boundary Graphs...")

features = ["sepal_length", "sepal_width"]

X_visual = df[features]
y_visual = df["species"]

class_names = sorted(y_visual.unique())
class_to_number = {
    name: index for index, name in enumerate(class_names)
}

y_numeric = y_visual.map(class_to_number)

x_min, x_max = X_visual.iloc[:, 0].min() - 0.5, X_visual.iloc[:, 0].max() + 0.5
y_min, y_max = X_visual.iloc[:, 1].min() - 0.5, X_visual.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

for name in models:

    visual_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]) if name == "Logistic Regression" else Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    visual_model.fit(X_visual, y_visual)

    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=features
    )

    predictions = visual_model.predict(grid)

    Z = np.array([
        class_to_number[item] for item in predictions
    ]).reshape(xx.shape)

    plt.figure(figsize=(9, 6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    plt.scatter(
        X_visual.iloc[:, 0],
        X_visual.iloc[:, 1],
        c=y_numeric,
        cmap="viridis",
        edgecolors="black"
    )

    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(f"Decision Boundary - {name}")

    plt.savefig(
        f"outputs/{name.lower().replace(' ', '_')}_decision_boundary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Decision Boundary Graphs Saved!")

# ==========================================
# 9. FINAL RESULT
# ==========================================

best_model = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]

print("\n" + "=" * 60)
print("                 FINAL PROJECT RESULT")
print("=" * 60)

print("Best Model:", best_model["Model"])
print("Best Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")
print("Best F1 Score:", round(best_model["F1 Score"], 4))

print("\nAll Outputs Saved in 'outputs' Folder.")

print("\nPROJECT EXECUTED SUCCESSFULLY!")
print("=" * 60)
