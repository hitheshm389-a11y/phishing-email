import os
import re
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure Flask uses the templates directory from this project folder
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
MODEL_PATH = os.path.join('model', 'model.joblib')
VECT_PATH = os.path.join('model', 'vectorizer.joblib')

# Load model and vectorizer at startup if available
model = None
vectorizer = None
if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)

# Simple indicators extraction for explanation
def extract_indicators(text):
    indicators = []
    lower = text.lower()
    # urgent language
    if re.search(r"\burgent\b|\bimmediately\b|\basap\b|\baction required\b", lower):
        indicators.append('Urgent or threatening language')
    # credential requests
    if re.search(r"\blogin\b|\bpassword\b|\bcredentials\b|\bverify your account\b|\bconfirm your identity\b", lower):
        indicators.append('Requests for credentials or verification')
    # suspicious links
    urls = re.findall(r'https?://[^\s]+', text)
    if urls:
        indicators.append(f'Suspicious link(s) found: {", ".join(urls)}')
    # suspicious sender-like phrases
    if re.search(r"\binvoice\b|\bbilling\b|\bsubscription\b|\bsecurity alert\b", lower):
        indicators.append('Business/financial spoofing language such as invoices or billing')
    # generic greeting
    if re.search(r"dear customer|dear user", lower):
        indicators.append('Generic greeting (e.g., "Dear Customer")')
    return indicators

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not found. Please run the training script first.'}), 500
    data = request.get_json()
    email = data.get('email', '')
    if not email.strip():
        return jsonify({'error': 'No email content provided.'}), 400

    # Preprocessing (the vectorizer handles lowercasing and stopwords if configured)
    X = vectorizer.transform([email])
    proba = model.predict_proba(X)[0]
    # assume binary [legitimate, phishing] depending on label encoding; we'll map by class name if available
    classes = list(model.classes_)
    # find phishing class index if labeled 'phishing' else take max for class 1
    if 'phishing' in classes:
        phish_idx = classes.index('phishing')
    else:
        phish_idx = 1 if len(classes) > 1 else 0
    phish_conf = float(proba[phish_idx])
    pred_label = 'Phishing' if phish_conf >= 0.5 else 'Legitimate'
    confidence_pct = round(phish_conf * 100, 2) if pred_label == 'Phishing' else round((1 - phish_conf) * 100, 2)

    indicators = extract_indicators(email)
    # Construct explanation
    explanation = ''
    if indicators:
        explanation += 'Indicators: ' + '; '.join(indicators) + '. '
    else:
        explanation += 'No clear textual indicators detected. '
    explanation += 'Model result maps to MITRE ATT&CK technique **T1566 (Phishing)** when suspicious.'

    return jsonify({
        'prediction': pred_label,
        'confidence': confidence_pct,
        'indicators': indicators,
        'explanation': explanation
    })

if __name__ == '__main__':
    # Run with debug disabled and a non-default port to avoid reloader parity issues
    app.run(debug=False, port=5001)
