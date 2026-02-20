import re
import tldextract

def extract_features(url):
    features = {}

    # Length of URL
    features['url_length'] = len(url)

    # Count dots
    features['dot_count'] = url.count('.')

    # Count digits
    features['digit_count'] = sum(c.isdigit() for c in url)

    # Special characters
    features['special_char_count'] = len(re.findall(r'[^\w]', url))

    # HTTPS presence
    features['has_https'] = 1 if 'https' in url else 0

    # Presence of @
    features['has_at'] = 1 if '@' in url else 0
    features['hyphen_count'] = url.count('-')
    features['underscore_count'] = url.count('_')
    features['slash_count'] = url.count('/')

    #suspicious domains
    suspicious_words = ['login', 'secure', 'update', 'bank', 'verify', 'account']
    features['suspicious_word'] = 1 if any(word in url.lower() for word in suspicious_words) else 0

     # IP address detection
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    ext = tldextract.extract(url)
    features['subdomain_count'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    features['hostname_length'] = len(ext.domain)


    # Number of subdomains
    ext = tldextract.extract(url)
    features['subdomain_count'] = len(ext.subdomain.split('.')) if ext.subdomain else 0

    return list(features.values())
