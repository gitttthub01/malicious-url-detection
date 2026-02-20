import pickle
from feature_engineering import extract_features

model = pickle.load(open("../models/malicious_url_model.pkl", "rb"))

url = input("Enter URL: ")

features = extract_features(url)
prediction = model.predict([features])

if prediction[0] == 1:
    print("⚠️ Malicious URL")
else:
    print("✅ Safe URL")
