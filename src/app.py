from flask import Flask, request, render_template
import pickle
from src.feature_engineering import extract_features
import tldextract
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from collections import defaultdict
import os
import time
import requests
import base64

app = Flask(__name__)

# ======================================================
# CONFIGURATION
# ======================================================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scans.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY")

ALPHA = 0.7  # ML weight
LOWER_THRESHOLD = 0.45
UPPER_THRESHOLD = 0.65
MAX_ALLOWED_LATENCY = 3.0  # allow API latency

# ======================================================
# DATABASE MODEL
# ======================================================

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
# LOAD MODEL (PIPELINE)
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "malicious_url_model.pkl")
)

print("Loading model from:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

# ======================================================
# VIRUSTOTAL FUNCTION
# ======================================================

def get_virustotal_score(url):

    if not VIRUSTOTAL_API_KEY:
        return 0

    try:
        url_bytes = url.encode("utf-8")
        url_id = base64.urlsafe_b64encode(url_bytes).decode().strip("=")

        headers = {"x-apikey": VIRUSTOTAL_API_KEY}
        vt_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"

        response = requests.get(vt_url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            stats = data["data"]["attributes"]["last_analysis_stats"]

            malicious_votes = stats.get("malicious", 0)
            total_votes = sum(stats.values())

            return malicious_votes / total_votes if total_votes > 0 else 0

        return 0

    except Exception:
        return 0
# ======================================================
# TRUSTED DOMAINS
# ======================================================

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

    url = request.form.get('url')

    if not url:
        return render_template("index.html",
                               result="Invalid Input",
                               status="warning")

    url = url.strip().lower()

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
        start_time = time.time()

        # ML Prediction
        features = extract_features(url)
        ml_prob = model.predict_proba([features])[0][1]

        # VirusTotal Score
        vt_score = get_virustotal_score(url)

        # Hybrid Score
        hybrid_score = ALPHA * ml_prob + (1 - ALPHA) * vt_score

        latency = round(time.time() - start_time, 4)

        if latency > MAX_ALLOWED_LATENCY:
            result = "⚠ System Busy - Try Again"
            confidence = 0.0
            status = "warning"

        else:
            confidence = round(
                hybrid_score if hybrid_score > 0.5 else 1 - hybrid_score,
                4
            )

            if hybrid_score >= UPPER_THRESHOLD:
                result = "Malicious URL Detected"
                status = "malicious"

            elif hybrid_score <= LOWER_THRESHOLD:
                result = "Safe URL"
                status = "safe"

            else:
                result = "⚠ Suspicious - Requires Verification"
                status = "warning"

    # ==================================================
    # SAVE TO DATABASE
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
# ENTRYPOINT (Render Compatible)
# ======================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    print("VirusTotal Key Loaded:", bool(VIRUSTOTAL_API_KEY))
print("Key Length:", len(VIRUSTOTAL_API_KEY) if VIRUSTOTAL_API_KEY else 0)