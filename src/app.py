from flask import Flask, request, render_template
import pickle
from .feature_engineering import extract_features
import tldextract
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from collections import defaultdict
import os
import time

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
    result = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    latency = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ======================================================
# LOAD XGBOOST MODEL
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "malicious_url_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Loaded model type:", type(model))

# ======================================================
# TRUSTED DOMAINS
# ======================================================

trusted_domains = [
    "google.com", "facebook.com", "youtube.com",
    "microsoft.com", "amazon.com", "linkedin.com"
]

# ======================================================
# FAIL-SAFE CONFIGURATION
# ======================================================

LOWER_THRESHOLD = 0.45
UPPER_THRESHOLD = 0.65
MAX_ALLOWED_LATENCY = 1.0  # seconds

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

    # ==================================================
    # WHITELIST CHECK
    # ==================================================

    if domain_only in trusted_domains:
        result = "Safe URL (Trusted Domain)"
        confidence = 1.0
        status = "safe"
        latency = 0.0

    else:
        # ==================================================
        # LATENCY MEASUREMENT
        # ==================================================

        start_time = time.time()

        features = extract_features(url)
        features_scaled = scaler.transform([features])

        # XGBoost probability prediction
        prob_malicious = model.predict_proba(features_scaled)[0][1]

        end_time = time.time()
        latency = round(end_time - start_time, 4)

        print("Malicious Probability:", prob_malicious)
        print("Latency:", latency)

        # ==================================================
        # LATENCY FAIL-SAFE
        # ==================================================

        if latency > MAX_ALLOWED_LATENCY:
            result = "⚠ System Busy - Please Try Again"
            confidence = 0.0
            status = "warning"

        else:
            confidence_score = abs(prob_malicious - 0.5) * 2

            # ==================================================
            # PROBABILITY DECISION LOGIC
            # ==================================================

            if prob_malicious >= UPPER_THRESHOLD:
                result = "Malicious URL Detected"
                confidence = prob_malicious
                status = "malicious"

            elif prob_malicious <= LOWER_THRESHOLD:
                result = "Safe URL"
                confidence = 1 - prob_malicious
                status = "safe"

            else:
                result = "⚠ Suspicious - Requires Further Verification"
                confidence = confidence_score
                status = "warning"

    # ==================================================
    # STORE RESULT
    # ==================================================

    new_scan = Scan(
        url=url,
        result=result,
        confidence=confidence,
        latency=latency
    )

    db.session.add(new_scan)
    db.session.commit()

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        status=status,
        domain_info=domain_info,
        latency=latency
    )


# ======================================================
# REPORTS
# ======================================================

@app.route('/reports')
def reports():
    scans = Scan.query.order_by(Scan.timestamp.desc()).all()
    return render_template("reports.html", scans=scans)


# ======================================================
# ANALYTICS
# ======================================================

@app.route('/analytics')
def analytics():

    total = Scan.query.count()
    malicious = Scan.query.filter(Scan.result.like("%Malicious%")).count()
    safe = total - malicious

    detection_rate = round((malicious / total) * 100, 2) if total > 0 else 0

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
# RENDER ENTRYPOINT
# ======================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)