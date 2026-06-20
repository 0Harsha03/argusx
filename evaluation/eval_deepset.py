import asyncio
import os
import json
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry
from app.detection.threat_scorer import ThreatScorer

def evaluate_config(pipeline, dataset, config_name, w_pattern, w_semantic, w_behavioral, w_anomaly):
    print(f"\n{'='*50}\nEvaluating Configuration: {config_name}\n{'='*50}")
    
    # Temporarily override weights in the scorer
    pipeline._threat_scorer._w_pattern = w_pattern
    pipeline._threat_scorer._w_semantic = w_semantic
    pipeline._threat_scorer._w_behavioral = w_behavioral
    pipeline._threat_scorer._w_anomaly = w_anomaly
    
    y_true = []
    y_pred = []
    
    fps = []
    fns = []
    
    layer_triggers = {'pattern': 0, 'semantic': 0, 'behavioral': 0, 'anomaly': 0}
    
    def run_eval():
        for i, item in enumerate(dataset):
            prompt = item['text']
            label = item['label'] # 0 = Benign, 1 = Injection
            
            # Run detection
            res = pipeline.analyze(prompt)
            
            # Zero out scores that are disabled for this config to see pure ablation
            # Wait, the pipeline runs all engines, but the scorer uses weights. 
            # If weight is 0, that layer contributes 0 to the final score.
            
            # Determine prediction
            if res['decision'] in ['FLAG', 'SANITIZE', 'BLOCK']:
                pred = 1
            else:
                pred = 0
                
            y_true.append(label)
            y_pred.append(pred)
            
            # Analyze layer triggers (only for Full Config to save time)
            if config_name == "Full ArgusX" and pred == 1 and label == 1:
                # Find which layer contributed most heavily
                contributions = {
                    'pattern': res['pattern_score'] * w_pattern,
                    'semantic': res['semantic_score'] * w_semantic,
                    'behavioral': res['behavioral_score'] * w_behavioral,
                    'anomaly': res['anomaly_score'] * w_anomaly
                }
                max_layer = max(contributions, key=contributions.get)
                layer_triggers[max_layer] += 1
                
            # Log FPs and FNs
            if label == 0 and pred == 1:
                fps.append({"prompt": prompt, "decision": res['decision'], "final_score": res['final_score']})
            elif label == 1 and pred == 0:
                fns.append({"prompt": prompt, "decision": res['decision'], "final_score": res['final_score']})
                
    run_eval()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "fps": fps, "fns": fns, "layer_triggers": layer_triggers
    }

def main():
    print("Loading DeepSet Prompt Injection dataset...")
    ds = load_dataset('deepset/prompt-injections')
    
    # Merge train and test to run the "entire" dataset as requested
    all_data = list(ds['train']) + list(ds['test'])
    print(f"Total samples loaded: {len(all_data)}")
    
    # Initialize registry and pipeline
    registry = ModelRegistry()
    registry.load_all()
    pipeline = DetectionPipeline(registry)
    
    results = {}
    
    # Configuration A: Pattern Only (Weight 1.0, others 0.0)
    results['Config A'] = evaluate_config(pipeline, all_data, "Pattern Only", 1.0, 0.0, 0.0, 0.0)
    
    # Configuration B: Pattern + Semantic (0.5, 0.5)
    results['Config B'] = evaluate_config(pipeline, all_data, "Pattern + Semantic", 0.5, 0.5, 0.0, 0.0)
    
    # Configuration C: Pattern + Semantic + Behavioral (0.34, 0.33, 0.33)
    results['Config C'] = evaluate_config(pipeline, all_data, "Pattern + Semantic + Behavioral", 0.34, 0.33, 0.33, 0.0)
    
    # Configuration D: Full ArgusX (Original config weights)
    # 0.30, 0.25, 0.30, 0.15
    results['Config D'] = evaluate_config(pipeline, all_data, "Full ArgusX", 0.30, 0.25, 0.30, 0.15)
    
    print("\n\n" + "="*50)
    print("ANALYSIS OF FULL ARGUSX (Config D)")
    print("="*50)
    
    full = results['Config D']
    print(f"\nFalse Positives (Total: {len(full['fps'])}):")
    for fp in full['fps'][:5]:
        print(f"  [Score: {fp['final_score']}] {fp['prompt'][:100]}...")
        
    print(f"\nFalse Negatives (Total: {len(full['fns'])}):")
    for fn in full['fns'][:5]:
        print(f"  [Score: {fn['final_score']}] {fn['prompt'][:100]}...")
        
    print("\nLayer-wise Contribution Statistics (For True Positives):")
    for k, v in full['layer_triggers'].items():
        print(f"  {k.capitalize()}: {v} triggers")
        
if __name__ == "__main__":
    main()
