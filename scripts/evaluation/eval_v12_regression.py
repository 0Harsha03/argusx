import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import urllib.request
import pandas as pd
from datasets import load_dataset as hf_load_dataset

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def evaluate_dataset(name, texts, labels, pipeline):
    preds = []
    routes = []
    confs = []
    db_exec = 0
    sbert_exec = 0
    rf_exec = 0
    lof_exec = 0
    ts_exec = 0
    pi_bypass = 0
    
    for t in texts:
        res = pipeline.analyze(t)
        
        # Binary prediction (1 for Malicious, 0 for Benign)
        # Assuming final_decision = True if decision != 'ALLOW'
        decision = res.get('decision', 'ALLOW')
        is_malicious = 1 if decision != 'ALLOW' else 0
        preds.append(is_malicious)
        
        # Route logic
        meta = res.get('route_metadata', {})
        route = res.get('selected_route', meta.get('selected_route', 'Unknown'))
        conf = meta.get('routing_confidence', 0.0)
        routes.append(route)
        confs.append(float(conf))
        
        # Engine checks
        if route == 'PromptInjectionRoute':
            # PromptInjectionRoute uses DB + RF, bypasses Scorer if final_decision=True
            db_exec += 1
            rf_exec += 1
            # LOF is actually in CyberThreatRoute primarily in v9/v12 depending on implementation, but PI route uses RF.
            if res.get('binary_decision', False) is not None:
                pi_bypass += 1
        elif route == 'CyberThreatRoute':
            sbert_exec += 1
            rf_exec += 1
            lof_exec += 1
            ts_exec += 1 # CyberThreatRoute relies on Scorer usually
            
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    
    acc = accuracy_score(labels_arr, preds_arr)
    prec = precision_score(labels_arr, preds_arr, zero_division=0)
    rec = recall_score(labels_arr, preds_arr, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    cm = confusion_matrix(labels_arr, preds_arr)
    if cm.size == 1:
        if labels_arr[0] == 1:
            tp = cm[0][0]; fp=0; tn=0; fn=0
        else:
            tp=0; fp=0; tn=cm[0][0]; fn=0
    else:
        tn, fp, fn, tp = cm.ravel()
        
    n = len(texts)
    pi_pct = sum(1 for r in routes if r == 'PromptInjectionRoute') / n * 100
    ct_pct = sum(1 for r in routes if r == 'CyberThreatRoute') / n * 100
    avg_conf = np.mean(confs)
    
    return {
        'name': name,
        'N': n,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': (tp, fp, tn, fn),
        'pi_pct': pi_pct, 'ct_pct': ct_pct, 'avg_conf': avg_conf,
        'db_pct': db_exec / n * 100,
        'sbert_pct': sbert_exec / n * 100,
        'rf_pct': rf_exec / n * 100,
        'lof_pct': lof_exec / n * 100,
        'ts_pct': ts_exec / n * 100,
        'bypass_pct': pi_bypass / n * 100
    }

def fetch_json(url):
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'ArgusX-Eval/7.0'})
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

def print_res(r):
    print(f"\n--- {r['name']} ({r['N']} samples) ---")
    if r['N'] == 0: return
    print(f"Accuracy:  {r['acc']:.4f}")
    print(f"Precision: {r['prec']:.4f}")
    print(f"Recall:    {r['rec']:.4f}")
    print(f"F1 Score:  {r['f1']:.4f}")
    print(f"Confusion: TP={r['cm'][0]} FP={r['cm'][1]} TN={r['cm'][2]} FN={r['cm'][3]}")
    print("Route Distribution:")
    print(f"  PromptInjectionRoute: {r['pi_pct']:.1f}%")
    print(f"  CyberThreatRoute:     {r['ct_pct']:.1f}%")
    print(f"  Avg Routing Conf:     {r['avg_conf']:.4f}")
    print("Detector Execution:")
    print(f"  DistilBERT: {r['db_pct']:.1f}%")
    print(f"  SBERT:      {r['sbert_pct']:.1f}%")
    print(f"  RF:         {r['rf_pct']:.1f}%")
    print(f"  LOF:        {r['lof_pct']:.1f}%")
    print(f"  ThreatScorer: {r['ts_pct']:.1f}%")
    print(f"  PI Bypassed Scorer: {r['bypass_pct']:.1f}%")

def main():
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)
    
    print("Evaluating DeepSet...")
    ds = hf_load_dataset('deepset/prompt-injections', split='test')
    texts = [r['text'] for r in ds if r['text']]
    labels = [r['label'] for r in ds if r['text']]
    r = evaluate_dataset("DeepSet", texts, labels, pipeline)
    print_res(r)
    
    print("Evaluating SPML...")
    df = pd.read_csv('data/pi_corpus/test.csv')
    texts = df['prompt'].tolist()
    labels = df['label'].tolist()
    r = evaluate_dataset("SPML", texts, labels, pipeline)
    print_res(r)
    
    _GH = 'https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets'
    
    print("Evaluating MITRE...")
    d = fetch_json(f'{_GH}/mitre/mitre_benchmark_100_per_category_with_augmentation.json')
    texts = [i.get('base_prompt', '') for i in d]
    texts = [t for t in texts if t]
    labels = [1] * len(texts)
    r = evaluate_dataset("MITRE", texts, labels, pipeline)
    print_res(r)
    
    print("Evaluating Interpreter Abuse...")
    d = fetch_json(f'{_GH}/interpreter/interpreter.json')
    texts = [i.get('mutated_prompt', '') for i in d]
    texts = [t for t in texts if t]
    labels = [1] * len(texts)
    r = evaluate_dataset("Interpreter", texts, labels, pipeline)
    print_res(r)
    
    print("Evaluating Spear Phishing...")
    d = fetch_json(f'{_GH}/spear_phishing/multiturn_phishing_challenges.json')
    texts = [extract_phish(i) for i in d]
    texts = [t for t in texts if t]
    labels = [1] * len(texts)
    r = evaluate_dataset("Phishing", texts, labels, pipeline)
    print_res(r)

if __name__ == '__main__':
    main()
