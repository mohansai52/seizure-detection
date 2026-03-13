import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1 Load dataset
# -----------------------------

df = pd.read_csv("seizure_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())


# -----------------------------
# 2 Exploratory Data Analysis
# -----------------------------

print("\nClass Distribution")
print(df["label"].value_counts())

sns.countplot(x="label", data=df)
plt.title("Seizure vs Non-Seizure Distribution")
plt.show()

sns.pairplot(df, hue="label")
plt.show()


# -----------------------------
# 3 Feature / Target Split
# -----------------------------

X = df[["heart_rate","spo2","temperature","vibration"]]
y = df["label"]


# -----------------------------
# 4 Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5 Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 6 Model Comparison
# -----------------------------

models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(name, "CV Accuracy:", scores.mean())


# -----------------------------
# 7 Hyperparameter Tuning
# -----------------------------

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[None,10,20]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


# -----------------------------
# 8 Model Evaluation
# -----------------------------

predictions = best_model.predict(X_test_scaled)

print("\nClassification Report")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

roc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:,1])
print("ROC AUC:", roc)


# -----------------------------
# 9 Feature Importance
# -----------------------------

importance = best_model.feature_importances_

features = X.columns

plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()


# -----------------------------
# 10 Save Model
# -----------------------------

pickle.dump(best_model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model saved successfully")
