import os
import json
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_local_advbench():
    data = []
    files = [
        ('benign.json', 0),
        ('prompt_injection.json', 1),
        ('jailbreaks.json', 1),
        ('credential_theft.json', 1),
        ('malware_generation.json', 1),
        ('system_extraction.json', 1)
    ]
    for file, label in files:
        path = os.path.join('evaluation', file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                for item in content:
                    data.append({'prompt': item['prompt'], 'label': label, 'source': 'advbench'})
    return data

def main():
    print("1. Loading datasets...")
    data = []
    
    # AdvBench
    adv_data = load_local_advbench()
    data.extend(adv_data)
    print(f" Loaded {len(adv_data)} samples from local AdvBench")
    
    # DeepSet
    ds_deepset = load_dataset('deepset/prompt-injections')
    for split in ['train', 'test']:
        for item in ds_deepset[split]:
            data.append({'prompt': item['text'], 'label': item['label'], 'source': 'deepset'})
    print(f" Loaded 662 samples from DeepSet")
    
    # SPML
    ds_spml = load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection', split='train')
    for item in ds_spml:
        data.append({'prompt': item['User Prompt'], 'label': item['Prompt injection'], 'source': 'spml'})
    print(f" Loaded {len(ds_spml)} samples from SPML")
    
    # 2. Merge and deduplicate
    df = pd.DataFrame(data)
    print(f" Total raw samples: {len(df)}")
    
    # Preprocessing (basic lowercase)
    df = df.dropna(subset=['prompt'])
    df['prompt_clean'] = df['prompt'].str.lower()
    df = df.drop_duplicates(subset=['prompt_clean'])
    print(f" Total deduplicated samples: {len(df)}")
    print(df['label'].value_counts())
    
    X = df['prompt_clean'].values
    y = df['label'].values
    
    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Engineering: Rebuild TF-IDF
    print("\n2. Training TfidfVectorizer...")
    # Increase n-grams and max_features to capture paraphrasing and multi-lingual structure better
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Retrain Behavioral Engine (Random Forest)
    print("3. Training Behavioral Engine (RandomForestClassifier)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # 6. Retrain Anomaly Engine (Isolation Forest)
    print("4. Training Anomaly Engine (IsolationForest)...")
    # Fit IF on benign training data to isolate structural anomalies
    benign_idx = (y_train == 0)
    if benign_idx.sum() > 0:
        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        iso_forest.fit(X_train_vec[benign_idx])
    else:
        print(" Warning: No benign data found for IF, fitting on all.")
        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        iso_forest.fit(X_train_vec)
    
    # 7. Evaluate Behavioral Engine Validation Performance
    print("\nBehavioral Engine Validation Performance:")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    
    # 8. Save Artifacts
    print("\n5. Saving Artifacts to app/models/artifacts/...")
    os.makedirs(os.path.join('app', 'models', 'artifacts'), exist_ok=True)
    
    with open(os.path.join('app', 'models', 'artifacts', 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join('app', 'models', 'artifacts', 'behavioral_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
        
    with open(os.path.join('app', 'models', 'artifacts', 'anomaly_detector.pkl'), 'wb') as f:
        pickle.dump(iso_forest, f)
        
    print("Training complete! v6 models deployed.")

if __name__ == "__main__":
    main()
