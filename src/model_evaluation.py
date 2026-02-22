import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from statsmodels.stats.contingency_tables import mcnemar

from feature_engineering import extract_features

# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("../dataset/final_dataset.csv")

X_raw = df["url"].apply(extract_features)
X = pd.DataFrame(X_raw.tolist())
y = df["label"]

# =====================================================
# 2. TRAIN TEST SPLIT (80/20)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 3. FEATURE SCALING (Z-score Normalization)
# z = (x - μ) / σ
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 4. TRAIN MODELS
# =====================================================

lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

lr.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

import pickle
import os

# Create models folder if not exists
os.makedirs("../models", exist_ok=True)

# Save XGBoost model
pickle.dump(xgb, open("../models/malicious_url_model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))

print("XGBoost model saved successfully.")

# =====================================================
# 5. PREDICTIONS
# =====================================================

lr_pred = lr.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)
xgb_pred = xgb.predict(X_test_scaled)

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# 6. PERFORMANCE REPORTS
# =====================================================

print("\n========== BASELINE MODELS ==========\n")

models = {
    "Logistic Regression": (lr_pred, lr_probs),
    "SVM": (svm_pred, svm_probs),
    "Random Forest": (rf_pred, rf_probs),
    "XGBoost": (xgb_pred, xgb_probs)
}

for name, (preds, probs) in models.items():
    print(f"=== {name} ===")
    print(classification_report(y_test, preds))
    print(f"{name} AUC:", roc_auc_score(y_test, probs))
    print("\n")

# =====================================================
# 7. ROC CURVE
# =====================================================

plt.figure(figsize=(8,6))

for name, (preds, probs) in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("ROC_curve.png")
plt.close()

# =====================================================
# 8. CONFUSION MATRICES
# =====================================================

for name, (preds, _) in models.items():
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# =====================================================
# 9. MCNEMAR SIGNIFICANCE TEST (RF vs XGB)
# =====================================================

correct_rf = (rf_pred == y_test).values
correct_xgb = (xgb_pred == y_test).values

table = [[0,0],[0,0]]

for i in range(len(y_test)):
    if correct_rf[i] and correct_xgb[i]:
        table[0][0] += 1
    elif correct_rf[i] and not correct_xgb[i]:
        table[0][1] += 1
    elif not correct_rf[i] and correct_xgb[i]:
        table[1][0] += 1
    else:
        table[1][1] += 1

result = mcnemar(table, exact=True)
print("McNemar p-value (RF vs XGB):", result.pvalue)

# =====================================================
# 10. LATENCY BENCHMARKING
# =====================================================

print("\n========== LATENCY (Inference Time) ==========\n")

for name, model in [("LR", lr), ("SVM", svm), ("RF", rf), ("XGB", xgb)]:
    start = time.time()
    model.predict(X_test_scaled)
    end = time.time()
    print(f"{name} latency: {round(end - start, 5)} seconds")

# =====================================================
# 11. PROPOSED HYBRID THREAT MODEL
# =====================================================

def lexical_score(row):
    url_length = row[0]
    digit_count = row[2]
    special_chars = row[3]
    subdomain_count = row[11]

    score = (
        0.25 * (url_length / 150) +
        0.25 * (digit_count / 20) +
        0.25 * (special_chars / 20) +
        0.25 * (subdomain_count / 5)
    )

    return min(score, 1)


def time_decay(days_old, lambda_decay=0.05):
    return math.exp(-lambda_decay * days_old)


np.random.seed(42)
simulated_days = np.random.randint(0, 30, size=len(X_test))

alpha = 0.65
beta = 0.25
gamma = 0.10

hybrid_scores = []

for i in range(len(X_test)):
    P_ml = rf_probs[i]
    L = lexical_score(X_test.iloc[i])
    T = time_decay(simulated_days[i])

    score = alpha*P_ml + beta*L + gamma*T
    hybrid_scores.append(score)

threshold = 0.6
hybrid_pred = [1 if s > threshold else 0 for s in hybrid_scores]

print("\n========== HYBRID MODEL ==========")
print(classification_report(y_test, hybrid_pred))
print("Hybrid Accuracy:", accuracy_score(y_test, hybrid_pred))

print("\nEvaluation Complete.")

# =====================================================
# 12. DECISION CURVE ANALYSIS
# =====================================================

def decision_curve(y_true, probs, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        preds = (probs >= pt).astype(int)
        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))

        nb = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return net_benefits


thresholds = np.linspace(0.01, 0.99, 50)

nb_rf = decision_curve(y_test.values, rf_probs, thresholds)
nb_xgb = decision_curve(y_test.values, xgb_probs, thresholds)

plt.figure(figsize=(8,6))
plt.plot(thresholds, nb_rf, label="Random Forest")
plt.plot(thresholds, nb_xgb, label="XGBoost")

plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', label="Treat None")

prevalence = np.mean(y_test)
treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
plt.plot(thresholds, treat_all, linestyle='--', label="Treat All")

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.savefig("Decision_Curve.png")
plt.close()