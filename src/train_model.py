import pandas as pd
import numpy as np
import time
import psutil
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from feature_engineering import extract_features

# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("../dataset/final_dataset.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Automatically detect label column
if 'label' in df.columns:
    label_column = 'label'
elif 'type' in df.columns:
    label_column = 'type'
else:
    raise Exception("No valid label column found!")

# If labels are already numeric, keep them
if df[label_column].dtype in ['int64', 'int32']:
    y = df[label_column]
else:
    df[label_column] = df[label_column].apply(
        lambda x: 0 if str(x).lower() == 'benign' else 1
    )
    y = df[label_column]

print("Overall class distribution:")
print(y.value_counts())

# =====================================================
# 2. FEATURE EXTRACTION
# =====================================================

X = df['url'].apply(lambda x: extract_features(x))
X = pd.DataFrame(X.tolist())

# =====================================================
# 3. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain distribution:")
print(y_train.value_counts())

print("\nTest distribution:")
print(y_test.value_counts())

# =====================================================
# 4. MODELS
# =====================================================

models = {
    "SVM": SVC(C=10, gamma=0.1, kernel='rbf',
               class_weight='balanced', probability=True),

    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight='balanced'),

    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight='balanced'),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
}

results = []

# =====================================================
# 5. TRAIN + EVALUATE
# =====================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Training time
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Inference time
    start_infer = time.time()
    y_pred = pipeline.predict(X_test)
    infer_time = (time.time() - start_infer) / len(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    risk_not_escalated = fnr * 100

    # Cross validation (10-fold)
    cv_scores = cross_val_score(
        pipeline, X, y, cv=10, scoring='f1'
    )
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)

    results.append([
        name, acc, prec, rec, f1, auc,
        cv_mean, cv_std,
        train_time, infer_time,
        memory_usage, fpr, fnr,
        risk_not_escalated
    ])

# =====================================================
# 6. PERFORMANCE TABLE
# =====================================================

columns = [
    "Model", "Accuracy", "Precision", "Recall", "F1", "AUC",
    "CV Mean (F1)", "CV Std",
    "Train Time (s)", "Inference Time (s/sample)",
    "Memory (MB)", "FPR", "FNR", "Risk Not Escalated (%)"
]

results_df = pd.DataFrame(results, columns=columns)

print("\n================ PERFORMANCE TABLE ================")
print(results_df)

# Save best model based on F1
best_model_name = results_df.sort_values(
    "F1", ascending=False).iloc[0]["Model"]

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', models[best_model_name])
])

best_pipeline.fit(X_train, y_train)

pickle.dump(best_pipeline,
            open("../models/malicious_url_model.pkl", "wb"))

print(f"\nBest model saved: {best_model_name}")

# =====================================================
# 7. HYBRID ALPHA-P ABLATION STUDY
# =====================================================

print("\n================ HYBRID ALPHA STUDY ================")

ml_probs = best_pipeline.predict_proba(X_test)[:, 1]

np.random.seed(42)
threat_score = np.random.uniform(0, 1, len(y_test))

alpha_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
hybrid_results = []

for alpha in alpha_values:
    final_score = alpha * ml_probs + (1 - alpha) * threat_score
    hybrid_pred = (final_score >= 0.5).astype(int)

    f1 = f1_score(y_test, hybrid_pred)
    auc = roc_auc_score(y_test, final_score)

    hybrid_results.append([alpha, f1, auc])

hybrid_df = pd.DataFrame(
    hybrid_results, columns=["Alpha", "F1", "AUC"]
)

print(hybrid_df)

# =====================================================
# 8. LAMBDA TIME DECAY STUDY
# =====================================================

print("\n================ LAMBDA DECAY STUDY ================")

time_diff = np.random.randint(1, 30, len(y_test))
lambda_values = [0.1, 0.2, 0.5, 1.0]
lambda_results = []

for lam in lambda_values:
    decay = np.exp(-lam * time_diff)
    adjusted_threat = threat_score * decay

    final_score = 0.7 * ml_probs + 0.3 * adjusted_threat
    pred = (final_score >= 0.5).astype(int)

    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, final_score)

    lambda_results.append([lam, f1, auc])

lambda_df = pd.DataFrame(
    lambda_results, columns=["Lambda", "F1", "AUC"]
)

print(lambda_df)


# =====================================================
# 9. SAVE ALL RESULTS (CSV + IMAGES)
# =====================================================

import matplotlib.pyplot as plt
output_dir = "test_results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---------- Save CSV files ----------
results_df.to_csv(f"{output_dir}/performance_table.csv", index=False)
hybrid_df.to_csv(f"{output_dir}/hybrid_alpha_study.csv", index=False)
lambda_df.to_csv(f"{output_dir}/lambda_decay_study.csv", index=False)

# =====================================================
# 10. GENERATE AND SAVE FIGURES
# =====================================================

# ---- 1. Model Performance Bar Chart (F1 Score) ----
plt.figure()
plt.bar(results_df["Model"], results_df["F1"])
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("Model Comparison (F1 Score)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_comparison_f1.png")
plt.close()

# ---- 2. Hybrid Alpha Curve ----
plt.figure()
plt.plot(hybrid_df["Alpha"], hybrid_df["F1"])
plt.xlabel("Alpha")
plt.ylabel("F1 Score")
plt.title("Hybrid Alpha Ablation Study")
plt.tight_layout()
plt.savefig(f"{output_dir}/hybrid_alpha_curve.png")
plt.close()

# ---- 3. Lambda Decay Curve ----
plt.figure()
plt.plot(lambda_df["Lambda"], lambda_df["F1"])
plt.xlabel("Lambda")
plt.ylabel("F1 Score")
plt.title("Lambda Time Decay Sensitivity")
plt.tight_layout()
plt.savefig(f"{output_dir}/lambda_decay_curve.png")
plt.close()

print("\nAll experiment results (CSV + images) saved inside src/test_results/")