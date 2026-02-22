import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from feature_engineering import extract_features

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("../dataset/final_dataset.csv")

X_raw = df["url"].apply(extract_features)
X = pd.DataFrame(X_raw.tolist())
y = df["label"]

# ==========================================
# 2. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. TRAIN MODELS (ONLY RF & XGB)
# ==========================================

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

rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# 4. DECISION CURVE ANALYSIS
# ==========================================

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

print("Decision Curve saved successfully.")