import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

pd.options.mode.chained_assignment = None

def load_internal_cyber():
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

def get_test_sets():
    # DeepSet
    ds_deepset = load_dataset('deepset/prompt-injections')
    ds_test = pd.DataFrame([{'prompt': i['text'], 'label': i['label']} for i in ds_deepset['test']])
    
    # SPML
    spml = load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection', split='train')
    spml_df = pd.DataFrame([{'prompt': i['User Prompt'], 'label': i['Prompt injection']} for i in spml]).sample(3000, random_state=42)
    _, _, spml_test = split_70_15_15(spml_df)
    
    # Local Data
    local_df = load_internal_cyber().dropna(subset=['prompt']).drop_duplicates(subset=['prompt'])
    internal_df, adv_df = train_test_split(local_df, test_size=0.5, random_state=42, stratify=local_df['label'])
    _, _, int_test = split_70_15_15(internal_df)
    _, _, adv_test = split_70_15_15(adv_df)
    
    return {
        "DeepSet Test": ds_test,
        "SPML Test": spml_test,
        "Internal Cyber Test": int_test,
        "AdvBench Test": adv_test
    }

def exp_a(r):
    return (r['ps'] * 0.30) + (r['ss'] * 0.25) + (r['bs'] * 0.30) + (r['ans'] * 0.15)

def exp_b(r):
    return (r['ps'] * 0.34) + (r['ss'] * 0.33) + (r['bs'] * 0.33)

def exp_c(r):
    score = (r['ps'] * 0.30) + (r['ss'] * 0.25) + (r['bs'] * 0.30)
    if r['ans'] > 0:
        score += r['ans'] * 0.15
    return score

def main():
    test_sets = get_test_sets()
    
    registry = ModelRegistry()
    registry.load_all() # Load v6.0 LODO models
    pipeline = DetectionPipeline(registry)
    
    thresholds = [30.0, 32.5, 35.0, 37.5, 40.0]
    experiments = {
        "A (Baseline)": exp_a,
        "B (Config C)": exp_b,
        "C (Recalibrated)": exp_c
    }
    
    all_results = {}
    
    for name, df in test_sets.items():
        print(f"\nProcessing {name} (N={len(df)})...")
        results = []
        for _, row in df.iterrows():
            res = pipeline.analyze(row['prompt'])
            r_dict = {
                'prompt': row['prompt'], 'label': row['label'],
                'ps': res['pattern_score'], 'ss': res['semantic_score'],
                'bs': res['behavioral_score'], 'ans': res['anomaly_score'],
                'fs_real': res['final_score'], 'dec_real': res['decision'],
                'critical_override': res['final_score'] >= 80.0 and res['pattern_score'] >= 80.0
            }
            results.append(r_dict)
            
        dataset_res = {}
        for exp_name, exp_func in experiments.items():
            dataset_res[exp_name] = {}
            for th in thresholds:
                y_true = []
                y_pred = []
                fns = []
                fps = []
                for r in results:
                    score = exp_func(r)
                    if r['critical_override']: score = max(score, 80.0)
                    pred = 1 if score >= th else 0
                    
                    y_true.append(r['label'])
                    y_pred.append(pred)
                    
                    if r['label'] == 1 and pred == 0:
                        fns.append((score, r))
                    elif r['label'] == 0 and pred == 1:
                        fps.append((score, r))
                        
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
                
                fns = sorted(fns, key=lambda x: x[0], reverse=True)[:50]
                fps = sorted(fps, key=lambda x: x[0], reverse=True)[:50]
                
                dataset_res[exp_name][th] = {
                    "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "fns": fns, "fps": fps
                }
        all_results[name] = dataset_res
        
    print("\n=================================================")
    print("THRESHOLD SENSITIVITY STUDY")
    print("=================================================")
    for dataset_name, exp_dict in all_results.items():
        print(f"\n--- {dataset_name} ---")
        for exp_name, th_dict in exp_dict.items():
            print(f"\nExperiment: {exp_name}")
            print(f"{'Thresh':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
            print("-" * 50)
            for th in thresholds:
                res = th_dict[th]
                print(f"{th:<8} | {res['acc']*100:>7.2f}% | {res['prec']*100:>7.2f}% | {res['rec']*100:>7.2f}% | {res['f1']*100:>7.2f}%")

    # The user asked for a comparison table at THRESHOLD = 35.0
    print("\n=================================================")
    print("COMPARISON TABLE (Threshold = 35.0)")
    print("=================================================")
    print(f"{'Dataset':<20} | {'Metric':<10} | {'v4.2.0':<8} | {'v6.0 (A)':<8} | {'v6.1-B':<8} | {'v6.1-C':<8}")
    print("-" * 75)
    
    # We hardcode the old v4.2.0 results we had for these datasets
    v4_results = {
        "DeepSet Test": {"acc": 61.21, "prec": 80.00, "rec": 33.33, "f1": 47.06},
        "SPML Test": {"acc": 89.56, "prec": 90.64, "rec": 96.58, "f1": 93.52},
        "Internal Cyber Test": {"acc": 95.65, "prec": 100.00, "rec": 94.74, "f1": 97.30},
        "AdvBench Test": {"acc": 86.96, "prec": 94.44, "rec": 89.47, "f1": 91.89}
    }
    
    for name in all_results.keys():
        for metric in ['acc', 'prec', 'rec', 'f1']:
            v4 = v4_results[name][metric]
            v6a = all_results[name]["A (Baseline)"][35.0][metric] * 100
            v61b = all_results[name]["B (Config C)"][35.0][metric] * 100
            v61c = all_results[name]["C (Recalibrated)"][35.0][metric] * 100
            print(f"{name:<20} | {metric.upper():<10} | {v4:>7.2f}% | {v6a:>7.2f}% | {v61b:>7.2f}% | {v61c:>7.2f}%")

    print("\n=================================================")
    print("FAILURE ANALYSIS (Threshold = 35.0) - Experiment C")
    print("=================================================")
    for name, exp_dict in all_results.items():
        res = exp_dict["C (Recalibrated)"][35.0]
        print(f"\n--- {name} ---")
        if res['fns']:
            print("Top FNs:")
            for score, r in res['fns'][:3]:
                print(f"[{score:.2f}] Pat:{r['ps']:.1f} Sem:{r['ss']:.1f} Beh:{r['bs']:.1f} Ans:{r['ans']:.1f} -> {r['prompt'][:60]}")
        if res['fps']:
            print("Top FPs:")
            for score, r in res['fps'][:3]:
                print(f"[{score:.2f}] Pat:{r['ps']:.1f} Sem:{r['ss']:.1f} Beh:{r['bs']:.1f} Ans:{r['ans']:.1f} -> {r['prompt'][:60]}")

if __name__ == "__main__":
    main()
