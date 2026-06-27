"""
Phase 4.2: SPML Regression Validation
Evaluates the v12 unified architecture on the SPML test split.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def main():
    print("Initializing pipeline...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)
    
    print("Loading data...")
    df = pd.read_csv('data/pi_corpus/test.csv')
    texts = df['prompt'].tolist()
    labels = df['label'].tolist()
    
    print(f"Loaded {len(texts)} samples from SPML test split.")
    
    preds = []
    diagnostics = []
    
    print("Evaluating...")
    for t in texts:
        res = pipeline.analyze(t)
        # 1 if malicious/blocked/flagged/sanitized
        # For PI, binary_decision usually captures this, but let's use the explicit route_metadata binary_decision or decision flag
        meta = res.get('route_metadata', {})
        decision = res.get('decision', 'ALLOW')
        is_malicious = 1 if decision != 'ALLOW' else 0
        preds.append(is_malicious)
        
        diagnostics.append({
            'prompt': t,
            'route': res.get('selected_route'),
            'meta': meta
        })
        
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    
    acc = accuracy_score(labels_arr, preds_arr)
    prec = precision_score(labels_arr, preds_arr, zero_division=0)
    rec = recall_score(labels_arr, preds_arr, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    
    tp = int(np.sum((preds_arr == 1) & (labels_arr == 1)))
    fp = int(np.sum((preds_arr == 1) & (labels_arr == 0)))
    tn = int(np.sum((preds_arr == 0) & (labels_arr == 0)))
    fn = int(np.sum((preds_arr == 0) & (labels_arr == 1)))
    
    print("\n--- Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    
    # Save diagnostics for error analysis if needed
    with open('results/v12_spml_diagnostics.json', 'w') as f:
        json.dump({'metrics': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, f)
        
if __name__ == '__main__':
    main()
