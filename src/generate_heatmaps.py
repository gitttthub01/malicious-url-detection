import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from feature_engineering import extract_features

# Load final dataset
df = pd.read_csv("../dataset/final_dataset.csv")

# Extract features
X = df["url"].apply(extract_features)
X = pd.DataFrame(X.tolist())

y = df["label"]

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
rf = RandomForestClassifier()

# Train
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)
rf_pred = rf.predict(X_test)

models = {
    "Logistic_Regression": lr_pred,
    "SVM": svm_pred,
    "Random_Forest": rf_pred
}

# Generate heatmaps
for name, preds in models.items():
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

print("Heatmaps saved successfully.")