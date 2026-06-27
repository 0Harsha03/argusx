import json, urllib.request, re, sys, random
from datasets import load_dataset as hf_load_dataset
from collections import Counter

pi_texts, cyber_texts = [], []
try:
    ds = hf_load_dataset('deepset/prompt-injections')
    for split in ds.keys(): pi_texts.extend([r['text'] for r in ds[split] if r['text']])
except: pass

try:
    ds_spml = hf_load_dataset('reshabhs/SPML_Chatbot_Prompt_Injection')
    pi_texts.extend([r['User Prompt'] for r in ds_spml[list(ds_spml.keys())[0]] if r['User Prompt']])
except: pass

_GH_BASE = 'https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets'
def fetch_json(path):
    req = urllib.request.Request(f'{_GH_BASE}/{path}', headers={'User-Agent': 'ArgusX/7.0'})
    with urllib.request.urlopen(req, timeout=30) as resp: return json.loads(resp.read().decode())

def extract_phish(row):
    for f in ('prompt', 'challenge', 'phishing_email', 'user_input', 'message', 'phishing_message'):
        if f in row and isinstance(row[f], str): return row[f]
    for f in ('attack_scenario', 'scenario'):
        if f in row and isinstance(row[f], dict):
            for nf in ('prompt', 'message', 'content'):
                if nf in row[f]: return str(row[f][nf])
    parts = [str(v) for v in row.values() if isinstance(v, str) and len(v)>30]
    return parts[0] if parts else None

subsets = [
    ('MITRE', 'mitre/mitre_benchmark_100_per_category_with_augmentation.json', 'base_prompt', False),
    ('MITRE FRR', 'mitre_frr/mitre_frr.json', 'mutated_prompt', False),
    ('Interpreter', 'interpreter/interpreter.json', 'mutated_prompt', False),
    ('Phishing', 'spear_phishing/multiturn_phishing_challenges.json', None, True),
]
for name, path, field, is_phish in subsets:
    try:
        raw = fetch_json(path)
        texts = [(extract_phish(i) if is_phish else i.get(field, '')) for i in raw]
        cyber_texts.extend([t.strip() for t in texts if t and isinstance(t, str)])
    except: pass

try:
    raw = fetch_json('prompt_injection/prompt_injection.json')
    texts = [i.get('user_input', '') for i in raw]
    pi_texts.extend([t.strip() for t in texts if t and isinstance(t, str)])
except: pass

with open('routing_study.txt', 'w', encoding='utf-8') as f:
    f.write(f'Prompt Route samples: {len(pi_texts)}\n')
    f.write(f'Cyber Route samples: {len(cyber_texts)}\n\n')
    
    random.seed(42)
    f.write('--- Semantic Separation (15 random each) ---\n')
    for t in random.sample(pi_texts, min(15, len(pi_texts))): f.write(f'PI: {repr(t[:100])}\n')
    for t in random.sample(cyber_texts, min(15, len(cyber_texts))): f.write(f'CYBER: {repr(t[:100])}\n')
    
    def get_vocab(texts):
        v = Counter()
        for t in texts:
            if t: v.update(re.findall(r'\b[a-z]{4,}\b', t.lower()))
        return set([w for w, c in v.items() if c > 10])
    
    pv = get_vocab(pi_texts)
    cv = get_vocab(cyber_texts)
    
    f.write('\n--- Vocabulary Overlap ---\n')
    f.write(f'Unique to PI (top 20): {list(pv - cv)[:20]}\n')
    f.write(f'Unique to Cyber (top 20): {list(cv - pv)[:20]}\n')
    f.write(f'Shared (top 20): {list(pv & cv)[:20]}\n')
    f.write(f'Overlap Jaccard Index: {len(pv & cv) / len(pv | cv):.4f}\n')
