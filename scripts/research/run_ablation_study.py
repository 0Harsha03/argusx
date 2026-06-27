import os
import sys
import time
import json
import urllib.request
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset as hf_load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.model_registry import ModelRegistry
from app.detection.pattern_detector import PatternDetector
from app.detection.behavioral_analyzer import BehavioralAnalyzer
from app.detection.anomaly_detector import AnomalyDetector
from app.detection.sbert_semantic_analyzer import SBERTSemanticAnalyzer
from app.detection.threat_scorer import ThreatScorer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert_pi"

_GH_BASE = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main"
    "/CybersecurityBenchmarks/datasets"
)

def fetch_json(path: str) -> list:
    url = f"{_GH_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArgusX-Eval/7.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def extract_phishing_prompt(row: dict) -> str:
    for field in ("prompt", "challenge", "phishing_email", "user_input", "message", "phishing_message"):
        if field in row and isinstance(row[field], str): return row[field]
    for field in ("attack_scenario", "scenario"):
        if field in row and isinstance(row[field], dict):
            nested = row[field]
            for nf in ("prompt", "message", "content"):
                if nf in nested: return str(nested[nf])
    parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 30]
    return parts[0] if parts else ""

def get_cyberseceval_data():
    datasets = []
    # mitre (malware/exploit)
    mitre_raw = fetch_json("mitre/mitre_benchmark_100_per_category_with_augmentation.json")
    datasets.extend([{"text": r.get("base_prompt", ""), "label": 1} for r in mitre_raw if isinstance(r.get("base_prompt"), str)])
    
    # spear phishing
    phish_raw = fetch_json("spear_phishing/multiturn_phishing_challenges.json")
    for r in phish_raw:
        prompt = extract_phishing_prompt(r)
        if prompt: datasets.append({"text": prompt, "label": 1})

    # mitre frr (benign)
    frr_raw = fetch_json("mitre_frr/mitre_frr.json")
    datasets.extend([{"text": r.get("mutated_prompt", ""), "label": 0} for r in frr_raw if isinstance(r.get("mutated_prompt"), str)])

    # interpreter abuse
    inter_raw = fetch_json("interpreter/interpreter.json")
    datasets.extend([{"text": r.get("mutated_prompt", ""), "label": 1} for r in inter_raw if isinstance(r.get("mutated_prompt"), str)])

    df = pd.DataFrame(datasets)
    df = df[df["text"].str.strip() != ""]
    return df

def get_deepset_data():
    ds = hf_load_dataset("deepset/prompt-injections")
    texts = [r["text"] for r in ds["test"]]
    labels = [int(r["label"]) for r in ds["test"]]
    return pd.DataFrame({"text": texts, "label": labels})

# Initialize models
registry = ModelRegistry()
registry.load_all()

pattern_detector = PatternDetector()
sbert_semantic = SBERTSemanticAnalyzer()
behavioral_analyzer = BehavioralAnalyzer(registry.behavioral_model, registry.vectorizer)
anomaly_detector = AnomalyDetector(registry.anomaly_detector, registry.vectorizer)
threat_scorer = ThreatScorer()

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
db_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
db_model.eval()

def run_pipeline(texts, labels, use_distilbert, use_rf, use_lof):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_time = time.time()
    
    y_true = []
    y_pred = []
    
    for i, text in enumerate(texts):
        y_true.append(labels[i])
        
        # 1. Pattern
        pat_res = pattern_detector.analyze(text)
        
        # 2. Semantic
        if use_distilbert:
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = db_model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().item()
            sem_score = p * 100.0
        else:
            sem_res = sbert_semantic.analyze(text)
            sem_score = sem_res.score
            
        # 3. RF
        if use_rf:
            beh_res = behavioral_analyzer.analyze(text, pat_res.score, list({d["category"] for d in pat_res.details}))
            beh_score = beh_res.score
            beh_flags = beh_res.behavioral_flags
        else:
            beh_score = 0.0
            beh_flags = []
            
        # 4. LOF
        if use_lof:
            ano_res = anomaly_detector.analyze(text)
            ano_score = ano_res.score
        else:
            ano_score = 0.0
            
        # 5. Scorer
        threat = threat_scorer.compute(
            pattern_score=pat_res.score,
            semantic_score=sem_score,
            behavioral_score=beh_score,
            anomaly_score=ano_score,
            matched_patterns=pat_res.matched_rules,
            behavioral_flags=beh_flags,
            top_category=pat_res.top_category,
            prompt=text
        )
        
        pred = 1 if threat.decision in ["BLOCK", "SANITIZE", "FLAG"] else 0
        y_pred.append(pred)
        
    end_time = time.time()
    end_mem = process.memory_info().rss
    
    elapsed = end_time - start_time
    mem_delta_mb = (end_mem - start_mem) / (1024 * 1024)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "elapsed_s": elapsed, "latency_ms": (elapsed / len(texts)) * 1000,
        "mem_mb": max(0, mem_delta_mb)
    }

def main():
    print("Loading data...")
    df_pi = get_deepset_data()
    df_cyber = get_cyberseceval_data()
    print(f"PI size: {len(df_pi)} | Cyber size: {len(df_cyber)}")
    
    configs = [
        ("A (All)", True, True),
        ("B (No RF)", False, True),
        ("C (No LOF)", True, False),
        ("D (No RF, No LOF)", False, False)
    ]
    
    results = {"PI": {}, "Cyber": {}}
    
    print("\n--- Running PI Pipeline (DistilBERT) ---")
    for name, use_rf, use_lof in configs:
        print(f"Running {name}...")
        res = run_pipeline(df_pi["text"].tolist(), df_pi["label"].tolist(), use_distilbert=True, use_rf=use_rf, use_lof=use_lof)
        results["PI"][name] = res
        
    print("\n--- Running Cyber Pipeline (SBERT) ---")
    for name, use_rf, use_lof in configs:
        print(f"Running {name}...")
        res = run_pipeline(df_cyber["text"].tolist(), df_cyber["label"].tolist(), use_distilbert=False, use_rf=use_rf, use_lof=use_lof)
        results["Cyber"][name] = res

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
