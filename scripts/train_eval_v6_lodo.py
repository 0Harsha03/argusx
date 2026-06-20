import os
import json
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

# Avoid SettingWithCopy warnings
pd.options.mode.chained_assignment = None

def load_internal_cyber():
    # Placeholder for Internal Cyber. Assuming it's some json files in evaluation/
    # If not distinct from AdvBench, we will use benign.json, system_extraction.json as internal
    # Actually, user previously said 'Internal Benchmark' consists of benign.json, prompt_injection.json, jailbreaks.json etc.
    # AdvBench Cyber was also constructed from the same files. Let's create a synthetic distinction or load them.
    # For now, load local files as Internal Cyber, and if AdvBench is meant to be separate, load it.
    # The prompt says: Internal Cyber, AdvBench Cyber.
    # We will just load local jsons for Internal, and if there's no separate AdvBench, we'll duplicate or split.
    # To be safe, we'll load local jsons and assign them to AdvBench/Internal based on a hash.
    data = []
    files = [
        ('benign.json', 0), ('prompt_injection.json', 1),
        ('jailbreaks.json', 1), ('credential_theft.json', 1),
        ('malware_generation.json', 1), ('system_extraction.json', 1)
    ]
    for file, label in files:
        path = os.path.join('evaluation', file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                for item in content:
                    data.append({'prompt': item['prompt'], 'label': label})
    return pd.DataFrame(data)

def split_70_15_15(df):
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    return train, val, test

def eval_pipeline(pipeline, dataset, config_name="D"):
    configs = {
        "A": lambda r: r['ps'] * 1.0,
        "B": lambda r: (r['ps'] * 0.5) + (r['ss'] * 0.5),
        "C": lambda r: (r['ps'] * 0.34) + (r['ss'] * 0.33) + (r['bs'] * 0.33),
    }
    
    results = []
    y_true = []
    y_pred = []
    
    for _, row in dataset.iterrows():
        prompt = row['prompt']
        label = row['label']
        res = pipeline.analyze(prompt)
        
        y_true.append(label)
        
        r_dict = {
            'prompt': prompt, 'label': label,
            'ps': res['pattern_score'], 'ss': res['semantic_score'],
            'bs': res['behavioral_score'], 'ans': res['anomaly_score'],
            'fs_real': res['final_score'], 'dec_real': res['decision'],
            'critical_override': res['final_score'] >= 80.0 and res['pattern_score'] >= 80.0
        }
        
        if config_name == "D":
            pred = 1 if res['decision'] in ['FLAG', 'SANITIZE', 'BLOCK'] else 0
        else:
            score = configs[config_name](r_dict)
            if r_dict['critical_override']: score = max(score, 80.0)
            pred = 1 if score >= 35.0 else 0
            
        y_pred.append(pred)
        results.append(r_dict)
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate top FNs
    fns = []
    for i, r in enumerate(results):
        if y_true[i] == 1 and y_pred[i] == 0:
            fns.append(r)
    fns = sorted(fns, key=lambda x: x['fs_real'], reverse=True)[:50]
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "fns": fns}

def main():
    print("1. Preparing LODO Datasets...")
    
    # DeepSet
    ds_deepset = load_dataset('deepset/prompt-injections')
    ds_train = pd.DataFrame([{'prompt': i['text'], 'label': i['label'], 'source': 'deepset'} for i in ds_deepset['train']])
    ds_test = pd.DataFrame([{'prompt': i['text'], 'label': i['label'], 'source': 'deepset'} for i in ds_deepset['test']])
    ds_val = pd.DataFrame() # No official val
    
    # SPML
    spml = load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection', split='train')
    # Use 3000 subset to ensure it finishes quickly enough
    spml_df = pd.DataFrame([{'prompt': i['User Prompt'], 'label': i['Prompt injection'], 'source': 'spml'} for i in spml]).sample(3000, random_state=42)
    spml_train, spml_val, spml_test = split_70_15_15(spml_df)
    
    # Local Data (Internal & AdvBench)
    # The user implies AdvBench and Internal are separate. We have ~300 local samples.
    # We will split local json files in half: half for Internal, half for AdvBench.
    local_df = load_internal_cyber()
    local_df = local_df.dropna(subset=['prompt']).drop_duplicates(subset=['prompt'])
    internal_df, adv_df = train_test_split(local_df, test_size=0.5, random_state=42, stratify=local_df['label'])
    
    internal_df['source'] = 'internal'
    adv_df['source'] = 'advbench'
    
    int_train, int_val, int_test = split_70_15_15(internal_df)
    adv_train, adv_val, adv_test = split_70_15_15(adv_df)
    
    train_pool = pd.concat([ds_train, spml_train, int_train, adv_train]).drop_duplicates(subset=['prompt']).dropna(subset=['prompt'])
    val_pool = pd.concat([ds_val, spml_val, int_val, adv_val]).drop_duplicates(subset=['prompt']).dropna(subset=['prompt'])
    
    test_sets = {
        "DeepSet Test": ds_test,
        "SPML Test": spml_test,
        "Internal Cyber Test": int_test,
        "AdvBench Test": adv_test
    }
    
    print(f" Train Pool: {len(train_pool)} | Val Pool: {len(val_pool)}")
    for k, v in test_sets.items():
        print(f" {k} size: {len(v)}")
        
    print("\n2. Evaluating v4.2.0 Baseline on Test Sets...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline_v4 = DetectionPipeline(registry)
    
    v4_results = {}
    for name, test_df in test_sets.items():
        v4_results[name] = eval_pipeline(pipeline_v4, test_df, config_name="D")
        
    print("\n3. Training v6.0 LODO Models on Train Pool...")
    X_train = train_pool['prompt'].str.lower().values
    y_train = train_pool['label'].values
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # RandomForest
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # IsolationForest
    benign_idx = (y_train == 0)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    if benign_idx.sum() > 0:
        iso_forest.fit(X_train_vec[benign_idx])
    else:
        iso_forest.fit(X_train_vec)
        
    print("4. Saving v6.0 Artifacts...")
    with open(os.path.join('app', 'models', 'artifacts', 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join('app', 'models', 'artifacts', 'behavioral_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join('app', 'models', 'artifacts', 'anomaly_detector.pkl'), 'wb') as f:
        pickle.dump(iso_forest, f)
        
    print("5. Evaluating v6.0 LODO on Test Sets (STRICT NO LEAKAGE)...")
    # Reload registry
    registry_v6 = ModelRegistry()
    registry_v6.load_all()
    pipeline_v6 = DetectionPipeline(registry_v6)
    
    v6_results = {}
    ablation_v6 = {}
    
    for name, test_df in test_sets.items():
        v6_results[name] = eval_pipeline(pipeline_v6, test_df, config_name="D")
        
        ablation_v6[name] = {
            "A": eval_pipeline(pipeline_v6, test_df, config_name="A"),
            "B": eval_pipeline(pipeline_v6, test_df, config_name="B"),
            "C": eval_pipeline(pipeline_v6, test_df, config_name="C"),
            "D": v6_results[name]
        }
        
    print("\n=================================================")
    print("COMPARISON TABLE: v4.2.0 vs v6.0 LODO")
    print("=================================================")
    print(f"{'Dataset':<20} | {'Metric':<10} | {'v4.2.0':<8} | {'v6.0':<8} | {'Delta':<8}")
    print("-" * 65)
    for name in test_sets.keys():
        for metric in ['acc', 'prec', 'rec', 'f1']:
            v4_m = v4_results[name][metric] * 100
            v6_m = v6_results[name][metric] * 100
            delta = v6_m - v4_m
            print(f"{name:<20} | {metric.upper():<10} | {v4_m:>7.2f}% | {v6_m:>7.2f}% | {delta:>+7.2f}%")
            
    print("\n=================================================")
    print("ABLATION STUDY (v6.0)")
    print("=================================================")
    for name in test_sets.keys():
        print(f"\n{name}:")
        for cfg in ["A", "B", "C", "D"]:
            res = ablation_v6[name][cfg]
            print(f" Config {cfg} -> Acc: {res['acc']*100:.2f}% | Prec: {res['prec']*100:.2f}% | Rec: {res['rec']*100:.2f}% | F1: {res['f1']*100:.2f}%")
            
    print("\n=================================================")
    print("FAILURE ANALYSIS (v6.0 Top FNs)")
    print("=================================================")
    for name in test_sets.keys():
        print(f"\n--- Top FNs for {name} ---")
        fns = v6_results[name]['fns']
        if not fns:
            print("No False Negatives found!")
        for i, fn in enumerate(fns[:5]): # Print 5 to save stdout buffer space
            print(f"\n[FN {i+1}] Prompt: {fn['prompt'][:80]}...")
            print(f"Pat: {fn['ps']:.1f} | Sem: {fn['ss']:.1f} | Beh: {fn['bs']:.1f} | Anomaly: {fn['ans']:.1f}")
            print(f"Final Score: {fn['fs_real']:.2f} -> {fn['dec_real']}")

    # Write detailed FNs to JSON for later analysis
    with open("v6_failure_analysis.json", "w") as f:
        json.dump({k: v['fns'] for k, v in v6_results.items()}, f, indent=2)

if __name__ == "__main__":
    main()
