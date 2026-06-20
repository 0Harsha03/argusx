import json
import logging
from datasets import load_dataset
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

DetectorFactory.seed = 0
logging.getLogger("app").setLevel(logging.WARNING)

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'

def evaluate_subset(pipeline, dataset, subset_name):
    y_true = []
    y_pred = []
    fns = []
    
    total_pattern = 0
    total_semantic = 0
    total_behavioral = 0
    total_anomaly = 0
    
    for item in dataset:
        prompt = item['text']
        label = item['label']
        
        res = pipeline.analyze(prompt)
        pred = 1 if res['decision'] in ['FLAG', 'SANITIZE', 'BLOCK'] else 0
        
        y_true.append(label)
        y_pred.append(pred)
        
        total_pattern += res['pattern_score']
        total_semantic += res['semantic_score']
        total_behavioral += res['behavioral_score']
        total_anomaly += res['anomaly_score']
        
        if label == 1 and pred == 0:
            fns.append({
                "prompt": prompt,
                "pattern": res['pattern_score'],
                "semantic": res['semantic_score'],
                "behavioral": res['behavioral_score'],
                "anomaly": res['anomaly_score'],
                "final_score": res['final_score'],
                "decision": res['decision']
            })
            
    n = len(dataset)
    if n > 0:
        avg_scores = {
            "pattern": total_pattern / n,
            "semantic": total_semantic / n,
            "behavioral": total_behavioral / n,
            "anomaly": total_anomaly / n
        }
    else:
        avg_scores = {}
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "name": subset_name,
        "n_samples": n,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "fns": fns,
        "avg_scores": avg_scores
    }

def main():
    print("Loading datasets and models...")
    ds = load_dataset('deepset/prompt-injections')
    all_data = list(ds['train']) + list(ds['test'])
    
    registry = ModelRegistry()
    registry.load_all()
    pipeline = DetectionPipeline(registry)
    
    english_subset = []
    german_subset = []
    other_subset = []
    
    print("Separating languages...")
    for item in all_data:
        lang = detect_language(item['text'])
        if lang == 'en':
            english_subset.append(item)
        elif lang == 'de':
            german_subset.append(item)
        else:
            other_subset.append(item)
            
    def print_dist(name, data):
        benign = sum(1 for d in data if d['label'] == 0)
        mal = sum(1 for d in data if d['label'] == 1)
        print(f"{name}: {len(data)} total (Benign: {benign}, Malicious: {mal})")
        
    print("\nLanguage Distribution:")
    print_dist("English", english_subset)
    print_dist("German", german_subset)
    print_dist("Other", other_subset)
    print_dist("Full Dataset", all_data)
    
    print("\nRunning Evaluation...")
    results_en = evaluate_subset(pipeline, english_subset, "English")
    results_de = evaluate_subset(pipeline, german_subset, "German")
    results_all = evaluate_subset(pipeline, all_data, "Full Dataset")
    
    for r in [results_en, results_de, results_all]:
        print(f"\n{'='*40}\nSubset: {r['name']}\n{'='*40}")
        print(f"Accuracy:  {r['acc']*100:.2f}%")
        print(f"Precision: {r['prec']*100:.2f}%")
        print(f"Recall:    {r['rec']*100:.2f}%")
        print(f"F1 Score:  {r['f1']*100:.2f}%")
        print("Confusion Matrix:")
        print(r['cm'])
        print(f"Avg Scores -> Pattern: {r['avg_scores'].get('pattern',0):.1f} | Sem: {r['avg_scores'].get('semantic',0):.1f} | Beh: {r['avg_scores'].get('behavioral',0):.1f} | Anomaly: {r['avg_scores'].get('anomaly',0):.1f}")
        
    print("\n" + "="*50)
    print("TOP 20 ENGLISH FALSE NEGATIVES")
    for i, fn in enumerate(results_en['fns'][:20]):
        print(f"\n[EN-FN {i+1}] {fn['prompt']}")
        print(f"Scores -> Pat:{fn['pattern']:.1f} Sem:{fn['semantic']:.1f} Beh:{fn['behavioral']:.1f} Anomaly:{fn['anomaly']:.1f} Final:{fn['final_score']:.2f} ({fn['decision']})")

    print("\n" + "="*50)
    print("TOP 20 GERMAN FALSE NEGATIVES")
    for i, fn in enumerate(results_de['fns'][:20]):
        print(f"\n[DE-FN {i+1}] {fn['prompt']}")
        print(f"Scores -> Pat:{fn['pattern']:.1f} Sem:{fn['semantic']:.1f} Beh:{fn['behavioral']:.1f} Anomaly:{fn['anomaly']:.1f} Final:{fn['final_score']:.2f} ({fn['decision']})")

if __name__ == "__main__":
    main()
