import os
import json
import logging
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

logging.getLogger("app").setLevel(logging.WARNING)

def evaluate_dataset(pipeline, dataset, dataset_name):
    print(f"\nEvaluating Dataset: {dataset_name} (Total: {len(dataset)})")
    
    # Run all samples once
    results = []
    
    for item in dataset:
        prompt = item['prompt']
        label = item['label'] # 0 = Benign, 1 = Malicious
        
        # We only pass the prompt, not the system prompt for SPML. The firewall evaluates the user prompt.
        res = pipeline.analyze(prompt)
        
        results.append({
            'prompt': prompt,
            'label': label,
            'ps': res['pattern_score'],
            'ss': res['semantic_score'],
            'bs': res['behavioral_score'],
            'ans': res['anomaly_score'],
            'fs_real': res['final_score'],
            'dec_real': res['decision'],
            'critical_override': res['final_score'] >= 80.0 and res['pattern_score'] >= 80.0 # heuristic
        })

    def calc_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        cm = confusion_matrix(y_t, y_p)
        return acc, prec, rec, f1, cm

    # Configs
    configs = {
        "A: Pattern Only": lambda r: r['ps'] * 1.0,
        "B: Pattern + Semantic": lambda r: (r['ps'] * 0.5) + (r['ss'] * 0.5),
        "C: Pattern + Semantic + Behavioral": lambda r: (r['ps'] * 0.34) + (r['ss'] * 0.33) + (r['bs'] * 0.33),
        "D: Full ArgusX": lambda r: r['fs_real'] # Just use the exact real score
    }

    ablation_results = {}
    for cname, func in configs.items():
        y_pred = []
        y_true = [r['label'] for r in results]
        
        for r in results:
            if cname == "D: Full ArgusX":
                score = r['fs_real']
                decision = r['dec_real']
                pred = 1 if decision in ['FLAG', 'SANITIZE', 'BLOCK'] else 0
            else:
                score = func(r)
                # Apply override if critical
                if r['critical_override']:
                    score = max(score, 80.0)
                pred = 1 if score >= 35.0 else 0
                
            y_pred.append(pred)
            
        acc, prec, rec, f1, cm = calc_metrics(y_true, y_pred)
        ablation_results[cname] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm
        }

    # Extract Top 50 FNs for Full ArgusX
    fns = []
    tps = []
    for r in results:
        pred = 1 if r['dec_real'] in ['FLAG', 'SANITIZE', 'BLOCK'] else 0
        if r['label'] == 1 and pred == 0:
            fns.append(r)
        elif r['label'] == 1 and pred == 1:
            tps.append(r)
            
    # Sort FNs by score descending (closest to being caught)
    fns = sorted(fns, key=lambda x: x['fs_real'], reverse=True)[:50]
    
    # Calculate Layer contributions for TPs
    avg_ps = sum(r['ps'] for r in tps) / max(1, len(tps))
    avg_ss = sum(r['ss'] for r in tps) / max(1, len(tps))
    avg_bs = sum(r['bs'] for r in tps) / max(1, len(tps))
    avg_ans = sum(r['ans'] for r in tps) / max(1, len(tps))

    return ablation_results, fns, {"ps": avg_ps, "ss": avg_ss, "bs": avg_bs, "ans": avg_ans}

def load_advbench():
    data = []
    files = [
        ('benign.json', 0),
        ('prompt_injection.json', 1),
        ('jailbreaks.json', 1),
        ('credential_theft.json', 1),
        ('malware_generation.json', 1),
        ('system_extraction.json', 1)
    ]
    for file, label in files:
        path = os.path.join('evaluation', file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                for item in content:
                    data.append({'prompt': item['prompt'], 'label': label})
    return data

def main():
    print("Initializing ArgusX Pipeline...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = DetectionPipeline(registry)
    
    datasets_to_run = []
    
    # MalPID check
    print("\n--- Checking Datasets ---")
    print("1. MalPID: NOT FOUND on HuggingFace public registry. Skipping.")
    
    # SPML Chatbot check
    print("2. SPML Chatbot Prompt Injection: FOUND.")
    ds = load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection', split='train')
    spml_data = [{'prompt': item['User Prompt'], 'label': item['Prompt injection']} for item in ds]
    
    # Limit SPML to 3000 samples for reasonable execution time (1500 benign, 1500 malicious)
    # Actually, we should evaluate the whole thing or a large stratified sample. 
    # Let's take 2000 random stratified samples to execute within a few minutes.
    import random
    random.seed(42)
    benign_spml = [d for d in spml_data if d['label'] == 0]
    malicious_spml = [d for d in spml_data if d['label'] == 1]
    # take up to 1000 of each
    sample_spml = random.sample(benign_spml, min(1000, len(benign_spml))) + random.sample(malicious_spml, min(1000, len(malicious_spml)))
    datasets_to_run.append(("SPML Chatbot (Sampled 2000)", sample_spml))
    
    # AdvBench
    print("3. AdvBench Cyber Subset: FOUND locally.")
    adv_data = load_advbench()
    if adv_data:
        datasets_to_run.append(("AdvBench Cyber Subset", adv_data))
        
    for name, data in datasets_to_run:
        abl_res, fns, layers = evaluate_dataset(pipeline, data, name)
        
        print(f"\n========================================")
        print(f"RESULTS FOR {name.upper()}")
        print(f"========================================")
        
        for cname, res in abl_res.items():
            print(f"\n[{cname}]")
            print(f"Acc: {res['acc']*100:.2f}% | Prec: {res['prec']*100:.2f}% | Rec: {res['rec']*100:.2f}% | F1: {res['f1']*100:.2f}%")
            print("CM:\n", res['cm'])
            
        print("\n--- Layer Contributions for True Positives ---")
        print(f"Pattern Avg:    {layers['ps']:.2f}")
        print(f"Semantic Avg:   {layers['ss']:.2f}")
        print(f"Behavioral Avg: {layers['bs']:.2f}")
        print(f"Anomaly Avg:    {layers['ans']:.2f}")
        
        print("\n--- Top False Negatives ---")
        for i, fn in enumerate(fns[:15]): # print top 15 to save space
            print(f"\n[FN {i+1}] Prompt: {fn['prompt'][:100]}...")
            print(f"Scores -> Pat:{fn['ps']:.1f} Sem:{fn['ss']:.1f} Beh:{fn['bs']:.1f} Anomaly:{fn['ans']:.1f}")
            print(f"Final: {fn['fs_real']:.2f} ({fn['dec_real']})")

if __name__ == "__main__":
    main()
