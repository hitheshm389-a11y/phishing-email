"""Train a TF-IDF + Logistic Regression phishing detector and save artifacts.

This script uses `data/emails.csv` (columns: text,label) shipped with the project.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = os.path.join('data', 'emails.csv')
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
VECT_PATH = os.path.join(MODEL_DIR, 'vectorizer.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA_PATH)
print(f'Dataset size: {len(df)}')

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_df=0.85, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print('Training Logistic Regression...')
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

print('Evaluating...')
y_pred = model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('Saving model and vectorizer...')
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECT_PATH)
print('Saved to', MODEL_PATH, 'and', VECT_PATH)
