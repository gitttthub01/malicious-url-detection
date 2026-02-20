import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from feature_engineering import extract_features

# Load dataset
df = pd.read_csv("../dataset/url_dataset.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Binary classification
df['type'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

# Feature extraction
X = df['url'].apply(extract_features)
X = pd.DataFrame(X.tolist())

y = df['type']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning
# Final SVM with best parameters
model = SVC(
    C=10,
    gamma=0.1,
    kernel='rbf',
    class_weight='balanced',
    probability=True
)

model.fit(X_train, y_train)


# Evaluation
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model AND scaler
pickle.dump(model, open("../models/malicious_url_model.pkl", "wb"))
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))
