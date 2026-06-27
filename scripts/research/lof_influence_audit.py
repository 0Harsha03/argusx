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

_GH_BASE = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets"

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

def get_datasets():
    datasets = {}
    
    # 1. DeepSet (PI)
    ds = hf_load_dataset("deepset/prompt-injections")
    datasets["DeepSet"] = [{"text": r["text"], "label": int(r["label"])} for r in ds["test"]]
    
    # 2. MITRE (Cyber)
    mitre_raw = fetch_json("mitre/mitre_benchmark_100_per_category_with_augmentation.json")
    datasets["MITRE"] = [{"text": r.get("base_prompt", ""), "label": 1} for r in mitre_raw if isinstance(r.get("base_prompt"), str)]
    
    # 3. Spear Phishing (Cyber)
    phish_raw = fetch_json("spear_phishing/multiturn_phishing_challenges.json")
    datasets["Spear Phishing"] = []
    for r in phish_raw:
        prompt = extract_phishing_prompt(r)
        if prompt: datasets["Spear Phishing"].append({"text": prompt, "label": 1})

    # 4. MITRE FRR (Cyber)
    frr_raw = fetch_json("mitre_frr/mitre_frr.json")
    datasets["MITRE FRR"] = [{"text": r.get("mutated_prompt", ""), "label": 0} for r in frr_raw if isinstance(r.get("mutated_prompt"), str)]

    # 5. Interpreter Abuse (Cyber)
    inter_raw = fetch_json("interpreter/interpreter.json")
    datasets["Interpreter Abuse"] = [{"text": r.get("mutated_prompt", ""), "label": 1} for r in inter_raw if isinstance(r.get("mutated_prompt"), str)]
    
    # 6. Prompt Injection Benchmark (PI)
    pi_raw = fetch_json("prompt_injection/prompt_injection.json")
    datasets["Blind PI"] = [{"text": r.get("user_input", ""), "label": 1} for r in pi_raw if isinstance(r.get("user_input"), str)]

    # Filter out empty prompts
    for k in datasets:
        datasets[k] = [d for d in datasets[k] if d["text"].strip()]
        
    return datasets

# Init models
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

def run_audit(dataset_name, data, use_distilbert):
    stats = {
        "lof_scores": [],
        "score_deltas": [],
        "decision_changed": 0,
        "decision_unchanged": 0,
        "threshold_crossings": [],
        "unique_recoveries": []
    }
    
    traces = []

    for i, item in enumerate(data):
        text = item["text"]
        label = item["label"]
        
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
        beh_res = behavioral_analyzer.analyze(text, pat_res.score, list({d["category"] for d in pat_res.details}))
        beh_score = beh_res.score
        beh_flags = beh_res.behavioral_flags
            
        # 4. LOF
        ano_res = anomaly_detector.analyze(text)
        ano_score = ano_res.score
        
        # 5. Scorer WITH LOF
        threat_with = threat_scorer.compute(
            pattern_score=pat_res.score,
            semantic_score=sem_score,
            behavioral_score=beh_score,
            anomaly_score=ano_score,
            matched_patterns=pat_res.matched_rules,
            behavioral_flags=beh_flags,
            top_category=pat_res.top_category,
            prompt=text
        )
        
        # 5. Scorer WITHOUT LOF
        threat_without = threat_scorer.compute(
            pattern_score=pat_res.score,
            semantic_score=sem_score,
            behavioral_score=beh_score,
            anomaly_score=0.0,
            matched_patterns=pat_res.matched_rules,
            behavioral_flags=beh_flags,
            top_category=pat_res.top_category,
            prompt=text
        )
        
        pred_with = 1 if threat_with.decision in ["BLOCK", "SANITIZE", "FLAG"] else 0
        pred_without = 1 if threat_without.decision in ["BLOCK", "SANITIZE", "FLAG"] else 0
        
        score_delta = threat_with.final_score - threat_without.final_score
        
        stats["lof_scores"].append(ano_score)
        stats["score_deltas"].append(score_delta)
        
        if threat_with.decision != threat_without.decision:
            stats["decision_changed"] += 1
            stats["threshold_crossings"].append(f"{threat_without.decision} -> {threat_with.decision}")
        else:
            stats["decision_unchanged"] += 1
            
        # Check for unique recovery
        # A recovery is when `pred_with == label` AND `pred_without != label`
        if pred_with == label and pred_without != label:
            rec_type = "False Positive" if label == 0 else "False Negative"
            stats["unique_recoveries"].append({
                "dataset": dataset_name,
                "index": i,
                "prompt": text[:150],
                "recovery_type": rec_type,
                "lof_score": ano_score,
                "threat_score_before": threat_without.final_score,
                "threat_score_after": threat_with.final_score,
                "decision_before": threat_without.decision,
                "decision_after": threat_with.decision
            })
            
    # Compute aggregates
    agg = {
        "dataset": dataset_name,
        "n_samples": len(data),
        "lof_min": float(np.min(stats["lof_scores"])),
        "lof_max": float(np.max(stats["lof_scores"])),
        "lof_mean": float(np.mean(stats["lof_scores"])),
        "lof_median": float(np.median(stats["lof_scores"])),
        "lof_std": float(np.std(stats["lof_scores"])),
        "delta_min": float(np.min(stats["score_deltas"])),
        "delta_max": float(np.max(stats["score_deltas"])),
        "delta_mean": float(np.mean(stats["score_deltas"])),
        "delta_median": float(np.median(stats["score_deltas"])),
        "delta_p95": float(np.percentile(stats["score_deltas"], 95)),
        "decision_unchanged": stats["decision_unchanged"],
        "decision_changed": stats["decision_changed"],
        "threshold_crossings": stats["threshold_crossings"],
        "unique_recoveries": stats["unique_recoveries"]
    }
    return agg

def main():
    print("Loading datasets...")
    datasets = get_datasets()
    
    # PI: DeepSet, Blind PI
    pi_datasets = ["DeepSet", "Blind PI"]
    cyber_datasets = ["MITRE", "Spear Phishing", "MITRE FRR", "Interpreter Abuse"]
    
    all_stats = []
    
    for d in pi_datasets:
        if d in datasets:
            print(f"Auditing {d} (PI) ...")
            agg = run_audit(d, datasets[d], use_distilbert=True)
            all_stats.append(agg)
            
    for d in cyber_datasets:
        if d in datasets:
            print(f"Auditing {d} (Cyber) ...")
            agg = run_audit(d, datasets[d], use_distilbert=False)
            all_stats.append(agg)

    with open("lof_influence_statistics.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
