import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
from app.services.model_registry import ModelRegistry
from app.routing.cyber_threat_route import CyberThreatRoute
from app.detection.pattern_detector import PatternDetector

def main():
    registry = ModelRegistry()
    registry.load_all()
    
    from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline
    pipeline = AdaptiveDetectionPipeline(registry)
    
    # Initialize routes and engines
    cyber_route = pipeline._router._cyber_route
    pattern_detector = PatternDetector()
    from app.detection.threat_scorer import ThreatScorer
    threat_scorer = ThreatScorer()
    
    df = pd.read_csv('cysecbench.csv')
    total = len(df)
    
    decisions = {'ALLOW': 0, 'FLAG': 0, 'SANITIZE': 0, 'BLOCK': 0}
    false_negatives = []
    cat_stats = {cat: {'total': 0, 'tp': 0, 'fn': 0} for cat in df['Category'].unique()}
    
    for i, row in df.iterrows():
        p = row['Prompt']
        cat = row['Category']
        
        # 1. Pattern
        pattern_result = pattern_detector.analyze(p)
        pattern_categories = list({d["category"] for d in pattern_result.details})
        
        # 2. Route (Oracle)
        res = cyber_route.process(p, p, pattern_result.score, pattern_categories)
        
        # 3. Score
        if res.final_decision:
            dec = res.enforcement_action or "ALLOW"
            final_score = res.final_score
        else:
            threat = threat_scorer.compute(
                pattern_score=res.pattern_score,
                semantic_score=res.semantic_score,
                behavioral_score=res.behavioral_score,
                anomaly_score=res.anomaly_score,
                matched_patterns=pattern_result.matched_rules,
                behavioral_flags=res.behavioral_flags,
                top_category=pattern_result.top_category,
                prompt=p,
            )
            dec = threat.decision
            final_score = threat.final_score
                
        if dec != 'ALLOW':
            cat_stats[cat]['tp'] += 1
        else:
            cat_stats[cat]['fn'] += 1
            false_negatives.append({
                'category': cat,
                'prompt': p,
                'semantic_score': res.semantic_score,
                'behavioral_score': res.behavioral_score,
                'pattern_score': res.pattern_score,
                'final_score': final_score,
                'decision': dec
            })
        
        decisions[dec] = decisions.get(dec, 0) + 1
        
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
    
    print("--- Task 2 Per-Category Detection ---")
    cat_res = []
    for cat, stats in cat_stats.items():
        if stats['total'] > 0:
            rate = stats['tp'] / stats['total'] * 100
            cat_res.append((cat, stats['total'], stats['tp'], stats['fn'], rate))
    
    cat_res.sort(key=lambda x: x[4], reverse=True)
    for r in cat_res:
        print(f"Category: {r[0]} | N: {r[1]} | TP: {r[2]} | FN: {r[3]} | Detection Rate: {r[4]:.2f}%")
    
    print("\n--- Task 3 Decision Distribution ---")
    for k, v in decisions.items():
        print(f"{k}: {v}")
        
    print("\n--- Task 4 False Negative Analysis ---")
    for fn_item in false_negatives[:100]:
        print(json.dumps(fn_item))

if __name__ == '__main__':
    main()
