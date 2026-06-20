import requests
from datasets import load_dataset
from app.detection.pattern_detector import PatternDetector

API_URL = "http://127.0.0.1:8000/api/v1/analyze"
detector = PatternDetector()

def analyze_prompt_local(prompt: str):
    """Hits the local PatternDetector directly to get the rule name."""
    res = detector.analyze(prompt)
    return res

def main():
    print("Loading JailbreakBench...")
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    jbb_prompts = []
    
    # We only care about the benign dataset for false positives
    for item in jbb["benign"]:
        jbb_prompts.append({"prompt": item["Goal"], "expected": "BENIGN"})
        
    fps = []
    
    for p in jbb_prompts:
        prompt = p["prompt"]
        
        # We only care about the cyber subset FPs.
        # Check if the prompt triggers one of the 3 new rules.
        res = analyze_prompt_local(prompt)
        matched_rules = set(res.matched_rules)
        
        target_rules = {"exploitation_generic", "data_theft_generic", "malware_generic"}
        intersection = matched_rules.intersection(target_rules)
        
        if intersection:
            # It fired one of the new rules on a benign prompt
            fps.append({
                "prompt": prompt,
                "rules": list(intersection)
            })
            
    print(f"\nFound {len(fps)} False Positives caused by the new generic rules:")
    for i, fp in enumerate(fps):
        print(f"\n{i+1}. PROMPT: {fp['prompt']}")
        print(f"   RULES:  {fp['rules']}")

if __name__ == "__main__":
    main()
