import json
import urllib.request
import re
import sys
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from datasets import load_dataset as hf_load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("Loading datasets...")

pi_texts = []
cyber_texts = []

# DeepSet
try:
    ds = hf_load_dataset('deepset/prompt-injections')
    for split in ds.keys():
        for r in ds[split]:
            if r['text']:
                pi_texts.append(r['text'])
except Exception as e:
    print('DeepSet err:', e)

# SPML
try:
    ds_spml = hf_load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection')
    split_key = list(ds_spml.keys())[0]
    for r in ds_spml[split_key]:
        if r['User Prompt']:
            pi_texts.append(r['User Prompt'])
except Exception as e:
    print('SPML err:', e)

_GH_BASE = 'https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets'
def fetch_json(path):
    req = urllib.request.Request(f'{_GH_BASE}/{path}', headers={'User-Agent': 'ArgusX/7.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def extract_phish(row):
    for f in ('prompt', 'challenge', 'phishing_email', 'user_input', 'message', 'phishing_message'):
        if f in row and isinstance(row[f], str): return row[f]
    for f in ('attack_scenario', 'scenario'):
        if f in row and isinstance(row[f], dict):
            for nf in ('prompt', 'message', 'content'):
                if nf in row[f]: return str(row[f][nf])
    parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 30]
    return parts[0] if parts else None

cyber_subsets = [
    ('MITRE', 'mitre/mitre_benchmark_100_per_category_with_augmentation.json', 'base_prompt', False),
    ('MITRE FRR', 'mitre_frr/mitre_frr.json', 'mutated_prompt', False),
    ('Interpreter', 'interpreter/interpreter.json', 'mutated_prompt', False),
    ('Phishing', 'spear_phishing/multiturn_phishing_challenges.json', None, True),
]
for name, path, field, is_phish in cyber_subsets:
    try:
        raw = fetch_json(path)
        texts = [(extract_phish(i) if is_phish else i.get(field, '')) for i in raw]
        cyber_texts.extend([t.strip() for t in texts if t and isinstance(t, str)])
    except Exception as e:
        print(name, 'err:', e)

try:
    raw = fetch_json('prompt_injection/prompt_injection.json')
    texts = [i.get('user_input', '') for i in raw]
    pi_texts.extend([t.strip() for t in texts if t and isinstance(t, str)])
except Exception as e:
    print('CyberSecEval PI err:', e)

# Create labels: 0 for Prompt, 1 for Cyber
all_texts = pi_texts + cyber_texts
# 0 = PROMPT, 1 = CYBER
all_labels = [0] * len(pi_texts) + [1] * len(cyber_texts)

print(f"Total PROMPT Route samples: {len(pi_texts)}")
print(f"Total CYBER Route samples: {len(cyber_texts)}")

X_train, X_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.20, random_state=42, stratify=all_labels)

print("Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression Classifier...")
clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
clf.fit(X_train_vec, y_train)

print("\n--- Evaluation ---")
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["PROMPT", "CYBER"]))

print("\n--- Misrouting Audit ---")
misrouted_count = 0
y_proba = clf.predict_proba(X_test_vec)

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        misrouted_count += 1
        gt_route = "CYBER" if y_test[i] == 1 else "PROMPT"
        pred_route = "CYBER" if y_pred[i] == 1 else "PROMPT"
        conf = y_proba[i][y_pred[i]]
        print(f"\nPrompt: {X_test[i][:200]}...")
        print(f"Ground Truth Route: {gt_route}")
        print(f"Predicted Route: {pred_route}")
        print(f"Prediction Confidence: {conf:.4f}")

print(f"\nTotal Misrouted: {misrouted_count}")

print("\n--- Feature Importance ---")
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]

# Top 30 for CYBER (positive coefs)
cyber_indices = np.argsort(coefs)[-30:][::-1]
cyber_features = [(feature_names[i], coefs[i]) for i in cyber_indices]

# Top 30 for PROMPT (negative coefs)
prompt_indices = np.argsort(coefs)[:30]
prompt_features = [(feature_names[i], coefs[i]) for i in prompt_indices]

print("\nTop 30 features indicating Cyber Route:")
for f, w in cyber_features:
    print(f"{f}: {w:.4f}")

print("\nTop 30 features indicating Prompt Route:")
for f, w in prompt_features:
    print(f"{f}: {w:.4f}")

print("\n--- Persistence ---")
with open('models/router_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('models/router_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Saved models/router_vectorizer.pkl and models/router_classifier.pkl")
