import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def main():
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)
    
    df = pd.read_csv('cysecbench.csv')
    total = len(df)
    
    results = []
    routes = []
    confs = []
    decisions = {'ALLOW': 0, 'FLAG': 0, 'SANITIZE': 0, 'BLOCK': 0}
    false_negatives = []
    
    cat_stats = {cat: {'total': 0, 'tp': 0, 'fn': 0} for cat in df['Category'].unique()}
    
    for i, row in df.iterrows():
        p = row['Prompt']
        cat = row['Category']
        
        # Analyze
        res = pipeline.analyze(p)
        dec = res.get('decision', 'ALLOW')
        meta = res.get('route_metadata', {})
        route = res.get('selected_route', meta.get('selected_route', 'Unknown'))
        conf = meta.get('routing_confidence', 0.0)
        
        routes.append(route)
        confs.append(float(conf))
        decisions[dec] = decisions.get(dec, 0) + 1
        
        cat_stats[cat]['total'] += 1
        if dec != 'ALLOW':
            cat_stats[cat]['tp'] += 1
        else:
            cat_stats[cat]['fn'] += 1
            false_negatives.append({
                'category': cat,
                'prompt': p,
                'route': route,
                'semantic_score': res.get('semantic_score', 0),
                'behavioral_score': res.get('behavioral_score', 0),
                'pattern_score': res.get('pattern_score', 0),
                'final_score': res.get('final_score', 0),
                'decision': dec
            })
            
    # Task 1 - Overall
    tp = sum(v['tp'] for v in cat_stats.values())
    fn = sum(v['fn'] for v in cat_stats.values())
    recall = tp / total * 100
    miss_rate = fn / total * 100
    
    print("--- Task 1 Overall Detection ---")
    print(f"Total prompts: {total}")
    print(f"Successfully detected (TP): {tp}")
    print(f"Missed (FN): {fn}")
    print(f"Detection Rate (Recall): {recall:.2f}%")
    print(f"Miss Rate: {miss_rate:.2f}%\n")
    
    # Task 2 - Per-Category
    print("--- Task 2 Per-Category Detection ---")
    cat_res = []
    for cat, stats in cat_stats.items():
        if stats['total'] > 0:
            rate = stats['tp'] / stats['total'] * 100
            cat_res.append((cat, stats['total'], stats['tp'], stats['fn'], rate))
    
    cat_res.sort(key=lambda x: x[4], reverse=True)
    for r in cat_res:
        print(f"Category: {r[0]} | N: {r[1]} | TP: {r[2]} | FN: {r[3]} | Detection Rate: {r[4]:.2f}%")
    
    # Task 3 - Routing Verification
    print("\n--- Task 3 Routing Verification ---")
    pi_pct = sum(1 for r in routes if r == 'PromptInjectionRoute') / total * 100
    ct_pct = sum(1 for r in routes if r == 'CyberThreatRoute') / total * 100
    avg_conf = np.mean(confs)
    print(f"PromptInjectionRoute: {pi_pct:.2f}%")
    print(f"CyberThreatRoute: {ct_pct:.2f}%")
    print(f"Average Routing Confidence: {avg_conf:.4f}\n")
    
    # Task 4 - Decision Distribution
    print("--- Task 4 Decision Distribution ---")
    for k, v in decisions.items():
        print(f"{k}: {v}")
        
    # Task 5 - False Negatives
    print("\n--- Task 5 False Negative Analysis ---")
    for fn_item in false_negatives[:100]: # limit to 100 in stdout to avoid massive logs, output all to a file if needed
        print(json.dumps(fn_item))

if __name__ == '__main__':
    main()
