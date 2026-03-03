import re
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    url = url.lower()
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    features = []

    # ------------------------------
    # 1. Length Based Features
    # ------------------------------
    features.append(len(url))                              # URL length
    features.append(len(parsed.netloc))                     # Hostname length
    features.append(len(parsed.path))                       # Path length

    # ------------------------------
    # 2. Character Counts
    # ------------------------------
    features.append(url.count('.'))                         # Dot count
    features.append(url.count('-'))                         # Hyphen count
    features.append(url.count('_'))                         # Underscore count
    features.append(url.count('/'))                         # Slash count
    features.append(url.count('?'))                         # Question mark count
    features.append(url.count('='))                         # Equal sign count
    features.append(url.count('@'))                         # @ symbol count

    # Digits
    features.append(sum(c.isdigit() for c in url))

    # Special characters ratio
    special_chars = len(re.findall(r'[^a-zA-Z0-9]', url))
    features.append(special_chars / len(url) if len(url) > 0 else 0)

    # ------------------------------
    # 3. Structural Features
    # ------------------------------
    features.append(1 if parsed.scheme == "https" else 0)  # HTTPS
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # IP address

    # Subdomain count
    features.append(len(ext.subdomain.split('.')) if ext.subdomain else 0)

    # TLD length
    features.append(len(ext.suffix))

    # ------------------------------
    # 4. Suspicious Keyword Features
    # ------------------------------
    suspicious_words = [
        'login', 'secure', 'update', 'bank', 'verify',
        'account', 'confirm', 'paypal', 'password',
        'signin', 'wp', 'admin'
    ]

    features.append(
        sum(word in url for word in suspicious_words)
    )

    # ------------------------------
    # 5. Entropy Feature (Important)
    # ------------------------------
    import math
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = -sum(p * math.log(p, 2) for p in prob)
    features.append(entropy)

    return features