from flask import Flask, request, render_template
import pickle
from .feature_engineering import extract_features
import tldextract
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from collections import defaultdict


app = Flask(__name__)

# ======================================================
# DATABASE CONFIGURATION
# ======================================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scans.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500))
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ======================================================
# LOAD TRAINED MODEL
# ======================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "malicious_url_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))


trusted_domains = [
    "google.com", "facebook.com", "youtube.com",
    "microsoft.com", "amazon.com", "linkedin.com"
]

# ======================================================
# ROUTES
# ======================================================

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url'].strip().lower()

    if not url.startswith("http"):
        url = "https://" + url

    ext = tldextract.extract(url)
    domain_only = f"{ext.domain}.{ext.suffix}"

    domain_info = {
        "Full URL": url,
        "Domain": ext.domain,
        "Subdomain": ext.subdomain if ext.subdomain else "None",
        "Top Level Domain": ext.suffix
    }

    # -----------------------
    # Prediction Logic
    # -----------------------
    if domain_only in trusted_domains:
        result = "Safe URL (Trusted Domain)"
        confidence = 1.0
        status = "safe"
    else:
        features = extract_features(url)
        features_scaled = scaler.transform([features])
        prob_malicious = model.predict_proba(features_scaled)[0][1]
        print("Malicious Probability:", prob_malicious)


        threshold = 0.7
        if prob_malicious > threshold:
            result = "Malicious URL Detected"
            confidence = prob_malicious
            status = "malicious"
        else:
            result = "Safe URL"
            confidence = 1 - prob_malicious
            status = "safe"

    # -----------------------
    # Save Scan to Database
    # -----------------------
    new_scan = Scan(url=url, result=result, confidence=confidence)
    db.session.add(new_scan)
    db.session.commit()

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        status=status,
        domain_info=domain_info
    )


# ======================================================
# REPORTS PAGE (Dynamic Table)
# ======================================================
@app.route('/reports')
def reports():
    scans = Scan.query.order_by(Scan.timestamp.desc()).all()
    return render_template("reports.html", scans=scans)


# ======================================================
# ANALYTICS DASHBOARD
# ======================================================
@app.route('/analytics')
def analytics():

    total = Scan.query.count()
    malicious = Scan.query.filter(Scan.result.like("%Malicious%")).count()
    safe = total - malicious

    detection_rate = round((malicious / total) * 100, 2) if total > 0 else 0

    # ----------- Last 7 Days Trend -----------
    last_7_days = defaultdict(int)

    for i in range(7):
        day = (datetime.utcnow() - timedelta(days=i)).date()
        last_7_days[str(day)] = 0

    scans = Scan.query.all()
    for scan in scans:
        day = str(scan.timestamp.date())
        if day in last_7_days:
            last_7_days[day] += 1

    dates = list(last_7_days.keys())
    counts = list(last_7_days.values())

    return render_template(
        "analytics.html",
        total=total,
        malicious=malicious,
        safe=safe,
        detection_rate=detection_rate,
        dates=dates,
        counts=counts
    )


# ======================================================
# MAIN
# ======================================================
if __name__ == '__main__':
      app.run(host="0.0.0.0", port=5000) 

