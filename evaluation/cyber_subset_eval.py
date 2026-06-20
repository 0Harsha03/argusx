import pandas as pd
import requests
import re
from datasets import load_dataset
from tabulate import tabulate

API_URL = "http://127.0.0.1:8000/api/v1/analyze"

CYBER_KEYWORDS = [
    r"\bhack(ing|er|ed)?\b", r"\bexploit(s|ing|ation)?\b", r"\bvulnerabilit(y|ies)\b",
    r"\bmalware\b", r"\bvirus(es)?\b", r"\bransomware\b", r"\bphishing\b",
    r"\bddos\b", r"\bsql injection\b", r"\bxss\b", r"\bbypass\b",
    r"\bunauthorized\b", r"\bsteal\b", r"\bcredential(s)?\b", r"\bpassword(s)?\b",
    r"\btoken(s)?\b", r"\bexfiltrate\b", r"\bdatabase\b", r"\bbotnet\b",
    r"\btrojan\b", r"\bbackdoor\b", r"\bjailbreak\b", r"\bprompt injection\b",
    r"\bsystem prompt\b", r"\binstructions\b", r"\bkeylogger\b", r"\brootkit\b",
    r"\bevasion\b", r"\bspoofing\b", r"\bcyber\b", r"\bpayload\b",
    r"\bcryptominer\b", r"\bkey\b", r"\bauth\b", r"\bsession\b", r"\bcookie\b",
    r"\bserver\b", r"\bnetwork\b", r"\brouter\b", r"\badmin\b", r"\blogin\b"
]
CYBER_PATTERN = re.compile("|".join(CYBER_KEYWORDS), re.IGNORECASE)

def is_cyber_related(prompt: str) -> bool:
    return bool(CYBER_PATTERN.search(prompt))

def analyze_prompt(prompt: str) -> str:
    try:
        resp = requests.post(API_URL, json={"prompt": prompt}, timeout=10)
        resp.raise_for_status()
        return resp.json()["decision"]
    except Exception as e:
        return "ERROR"

def run_dataset_eval(name: str, prompts: list):
    tp = tn = fp = fn = 0
    false_negatives = []
    
    for p in prompts:
        exp = p["expected"]
        actual = analyze_prompt(p["prompt"])
        if actual == "ERROR":
            continue
            
        is_threat = (actual in ["BLOCK", "SANITIZE", "FLAG"])
        if exp == "THREAT":
            if is_threat:
                tp += 1
            else:
                fn += 1
                false_negatives.append(p["prompt"])
        else:
            if not is_threat:
                tn += 1
            else:
                fp += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Dataset": name,
        "Total Samples": total,
        "Accuracy": f"{accuracy*100:.2f}%",
        "Precision": f"{precision*100:.2f}%",
        "Recall": f"{recall*100:.2f}%",
        "F1": f"{f1*100:.2f}%",
        "False Negatives": false_negatives
    }

def main():
    print("Loading AdvBench...")
    adv_df = pd.read_csv("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")
    adv_prompts = [{"prompt": str(row["goal"]), "expected": "THREAT"} for _, row in adv_df.iterrows()]
    adv_cyber = [p for p in adv_prompts if is_cyber_related(p["prompt"])]
    
    print("Loading JailbreakBench...")
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    jbb_prompts = []
    for item in jbb["harmful"]:
        jbb_prompts.append({"prompt": item["Goal"], "expected": "THREAT"})
    for item in jbb["benign"]:
        jbb_prompts.append({"prompt": item["Goal"], "expected": "BENIGN"})
    
    jbb_cyber = []
    for p in jbb_prompts:
        if p["expected"] == "BENIGN":
            jbb_cyber.append(p)
        elif is_cyber_related(p["prompt"]):
            jbb_cyber.append(p)
            
    print(f"AdvBench Cyber Subset: {len(adv_cyber)} / {len(adv_prompts)}")
    print(f"JailbreakBench Cyber Subset: {len(jbb_cyber)} / {len(jbb_prompts)}")
    
    results = []
    all_fns = {}
    
    for name, subset in [("AdvBench Cyber", adv_cyber), ("JailbreakBench Cyber", jbb_cyber)]:
        m = run_dataset_eval(name, subset)
        all_fns[name] = m.pop("False Negatives")
        results.append(m)
        
    print("\n=== METRICS ===")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    print("\n=== FALSE NEGATIVES ===")
    for name, fns in all_fns.items():
        print(f"\n{name} ({len(fns)} FNs):")
        for i, fn in enumerate(fns):
            print(f"  {i+1}. {fn}")

if __name__ == "__main__":
    main()
