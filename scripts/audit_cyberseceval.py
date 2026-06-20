"""
ArgusX v7 — CyberSecEval Rigorous Audit Script
================================================
Collects exact data for all 7 audit sections without retraining or modifying any model.

Run:
    $env:PYTHONPATH = "."
    python scripts/audit_cyberseceval.py > audit_output.txt
"""

import json, urllib.request, time
from collections import Counter, defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from app.services.model_registry import ModelRegistry
from app.services.sbert_detection_pipeline import SBERTDetectionPipeline
from app.services.detection_pipeline import DetectionPipeline

_GH_BASE = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets"
POSITIVE = {"FLAG", "SANITIZE", "BLOCK"}


def fetch(path):
    req = urllib.request.Request(f"{_GH_BASE}/{path}", headers={"User-Agent": "ArgusX-Audit/7.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def analyze_all(pipeline, prompts, label, name):
    results = []
    for p in prompts:
        if not p or not isinstance(p, str): continue
        res = pipeline.analyze(p.strip())
        pred = 1 if res["decision"] in POSITIVE else 0
        results.append({
            "prompt":   p.strip(),
            "label":    label,
            "ps":       res["pattern_score"],
            "ss":       res["semantic_score"],
            "bs":       res["behavioral_score"],
            "ans":      res["anomaly_score"],
            "fs":       res["final_score"],
            "dec":      res["decision"],
            "pred":     pred,
            "matched":  res.get("matched_patterns", []),
        })
    return results


def engine_contribution(results):
    """Determine primary engine responsible for each TP."""
    pattern_driven = semantic_driven = behavioral_driven = mixed = 0
    for r in results:
        if r["pred"] == 0 or r["label"] != 1:
            continue
        ps, ss, bs = r["ps"], r["ss"], r["bs"]
        if ps >= 70:
            pattern_driven += 1
        elif bs >= 70 and ss >= 30:
            mixed += 1
        elif bs >= 70:
            behavioral_driven += 1
        else:
            semantic_driven += 1
    total = pattern_driven + semantic_driven + behavioral_driven + mixed
    if total == 0:
        return {}
    return {
        "Pattern Engine":     round(pattern_driven / total * 100, 1),
        "Behavioral Engine":  round(behavioral_driven / total * 100, 1),
        "SBERT Semantic":     round(semantic_driven / total * 100, 1),
        "Mixed (Beh+Sem)":    round(mixed / total * 100, 1),
        "total_tp":           total,
    }


def main():
    print("=" * 70)
    print("ArgusX v7 CyberSecEval Rigorous Audit")
    print("=" * 70)

    # Init
    registry = ModelRegistry()
    registry.load_all()
    pipeline_v7 = SBERTDetectionPipeline(registry)

    # ── Load all subsets ────────────────────────────────────────────────────
    print("\nLoading subsets from GitHub...", flush=True)
    mitre_raw      = fetch("mitre/mitre_benchmark_100_per_category_with_augmentation.json")
    frr_raw        = fetch("mitre_frr/mitre_frr.json")
    phishing_raw   = fetch("spear_phishing/multiturn_phishing_challenges.json")
    injection_raw  = fetch("prompt_injection/prompt_injection.json")
    interp_raw     = fetch("interpreter/interpreter.json")

    mitre_prompts    = [r["base_prompt"] for r in mitre_raw if r.get("base_prompt")]
    frr_prompts      = [r["mutated_prompt"] for r in frr_raw if r.get("mutated_prompt")]
    inject_prompts   = [r["user_input"] for r in injection_raw if r.get("user_input")]
    interp_prompts   = [r["mutated_prompt"] for r in interp_raw if r.get("mutated_prompt")]
    # Phishing: extract any string field > 30 chars
    phishing_prompts = []
    for row in phishing_raw:
        parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 30]
        if parts:
            phishing_prompts.append(parts[0])

    print(f"  MITRE:          {len(mitre_prompts)} prompts")
    print(f"  MITRE FRR:      {len(frr_prompts)} prompts")
    print(f"  Phishing:       {len(phishing_prompts)} prompts")
    print(f"  Inj:            {len(inject_prompts)} prompts")
    print(f"  Interpreter:    {len(interp_prompts)} prompts")

    # ── Run evaluations ─────────────────────────────────────────────────────
    print("\nRunning evaluations...", flush=True)
    mitre_res    = analyze_all(pipeline_v7, mitre_prompts,    1, "MITRE")
    frr_res      = analyze_all(pipeline_v7, frr_prompts,      0, "MITRE FRR")
    phish_res    = analyze_all(pipeline_v7, phishing_prompts, 1, "Phishing")
    inject_res   = analyze_all(pipeline_v7, inject_prompts,   1, "Injection")
    interp_res   = analyze_all(pipeline_v7, interp_prompts,   1, "Interpreter")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B — LABELING AUDIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SECTION B — LABELING AUDIT")
    print("=" * 70)

    inj_cats = Counter(r.get("injection_type","?") for r in injection_raw)
    inj_risk = Counter(r.get("risk_category","?") for r in injection_raw)
    inj_langs = Counter(r.get("speaking_language","?") for r in injection_raw)
    frr_attack_types = Counter(r.get("attack_type","?") for r in frr_raw[:50])

    print(f"\nPrompt Injection — injection_type breakdown: {dict(inj_cats)}")
    print(f"Prompt Injection — risk_category breakdown:  {dict(inj_risk)}")
    print(f"Prompt Injection — language breakdown:       {dict(inj_langs)}")
    print(f"\nMITRE FRR — sample attack_type values: {dict(frr_attack_types)}")
    print(f"\nMITRE FRR is_malicious field: { {str(r.get('is_malicious',None)) for r in frr_raw[:10]} }")

    discarded_inject = sum(1 for r in injection_raw if not r.get("user_input"))
    print(f"\nPrompt Injection — prompts discarded (empty user_input): {discarded_inject}")
    print(f"Prompt Injection — total raw rows: {len(injection_raw)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C — METRIC AUDIT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SECTION C — METRIC AUDIT")
    print("=" * 70)

    def metrics_and_cm(res):
        yt = [r["label"] for r in res]
        yp = [r["pred"]  for r in res]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc  = (tp+tn)/(tp+tn+fp+fn)
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        fpr  = fp/(fp+tn) if (fp+tn) > 0 else 0
        fnr  = fn/(fn+tp) if (fn+tp) > 0 else 0
        return {"TP":tp,"FP":fp,"TN":tn,"FN":fn,
                "Acc":acc,"Prec":prec,"Rec":rec,"F1":f1,"FPR":fpr,"FNR":fnr}

    for name, res in [("MITRE", mitre_res), ("Phishing", phish_res), ("FRR", frr_res),
                      ("Injection", inject_res), ("Interpreter", interp_res)]:
        m = metrics_and_cm(res)
        print(f"\n{name} (N={len(res)}):")
        print(f"  TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
        print(f"  Acc={m['Acc']*100:.2f}% Prec={m['Prec']*100:.2f}% "
              f"Rec={m['Rec']*100:.2f}% F1={m['F1']*100:.2f}%")
        print(f"  FPR={m['FPR']*100:.2f}% FNR={m['FNR']*100:.2f}%")

    # SECTION C-15: MITRE_FRR decision breakdown
    print("\nMITRE FRR — Decision Breakdown:")
    frr_dec = Counter(r["dec"] for r in frr_res)
    for dec, cnt in sorted(frr_dec.items(), key=lambda x: -x[1]):
        pct = cnt / len(frr_res) * 100
        print(f"  {dec:<12} = {cnt:4d}  ({pct:.2f}%)")

    # SECTION C-16: FRR recomputed with BLOCK only
    frr_block_only = [1 if r["dec"] == "BLOCK" else 0 for r in frr_res]
    frr_block_count = sum(frr_block_only)
    print(f"\nFRR (BLOCK only): {frr_block_count}/{len(frr_res)} = {frr_block_count/len(frr_res)*100:.2f}%")
    print(f"FRR (FLAG+SANITIZE+BLOCK): {frr_dec.get('FLAG',0)+frr_dec.get('SANITIZE',0)+frr_dec.get('BLOCK',0)}/{len(frr_res)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D — RESULT PLAUSIBILITY / ENGINE CONTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SECTION D — ENGINE CONTRIBUTION ANALYSIS")
    print("=" * 70)

    for name, res in [("MITRE", mitre_res), ("Phishing", phish_res),
                      ("Injection", inject_res), ("Interpreter", interp_res)]:
        ec = engine_contribution(res)
        if ec:
            print(f"\n{name} TPs (N={ec.pop('total_tp')}) — Primary Engine:")
            for eng, pct in sorted(ec.items(), key=lambda x: -x[1]):
                print(f"  {eng:<25}: {pct:.1f}%")

    # Per-engine score distributions for MITRE TPs
    mitre_tps = [r for r in mitre_res if r["pred"] == 1 and r["label"] == 1]
    avg_ps = sum(r["ps"] for r in mitre_tps) / len(mitre_tps)
    avg_ss = sum(r["ss"] for r in mitre_tps) / len(mitre_tps)
    avg_bs = sum(r["bs"] for r in mitre_tps) / len(mitre_tps)
    avg_fs = sum(r["fs"] for r in mitre_tps) / len(mitre_tps)
    print(f"\nMITRE TPs Average Scores: Pat={avg_ps:.1f} Sem={avg_ss:.1f} Beh={avg_bs:.1f} Final={avg_fs:.1f}")

    # Pattern-only simulation (how many TPs have pat >= 80?)
    pat_driven_mitre = sum(1 for r in mitre_tps if r["ps"] >= 80)
    print(f"MITRE TPs with Pattern>=80: {pat_driven_mitre}/{len(mitre_tps)} = {pat_driven_mitre/len(mitre_tps)*100:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E — FALSE POSITIVE AUDIT (MITRE FRR top 25 FPs)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SECTION E — MITRE FRR: Top 25 False Positives")
    print("=" * 70)

    fps_frr = sorted([r for r in frr_res if r["pred"] == 1], key=lambda x: -x["fs"])[:25]
    print(f"\nTotal FPs: {len([r for r in frr_res if r['pred']==1])} / {len(frr_res)}")
    print(f"\n{'#':<4} {'Dec':<10} {'Pat':>5} {'Sem':>5} {'Beh':>5} {'FS':>6}  Prompt (first 90 chars)")
    print("-" * 120)
    for i, r in enumerate(fps_frr, 1):
        snippet = r["prompt"][:90].replace("\n", " ")
        rules = ", ".join(r["matched"][:2]) if r["matched"] else "no-pattern"
        print(f"{i:<4} {r['dec']:<10} {r['ps']:>5.1f} {r['ss']:>5.1f} {r['bs']:>5.1f} "
              f"{r['fs']:>6.2f}  {snippet}")
        print(f"      Triggered: {rules}")

    # Categorize FRR FPs
    cat_counts = defaultdict(int)
    for r in fps_frr:
        p = r["prompt"].lower()
        if any(w in p for w in ["detect", "prevent", "defense", "protect", "defensive", "block unauthorized", "firewall"]):
            cat_counts["Defensive Security Query"] += 1
        elif any(w in p for w in ["buffer overflow", "sql injection", "vulnerability", "cve-", "exploit", "xss", "ssrf"]):
            cat_counts["Educational Security Query"] += 1
        elif any(w in p for w in ["generate", "write", "implement", "code", "script", "program"]):
            cat_counts["Secure Coding Query"] += 1
        elif any(w in p for w in ["malware", "virus", "ransomware"]):
            cat_counts["Malware Analysis Query"] += 1
        else:
            cat_counts["Other"] += 1
    print(f"\nFRR FP Categories: {dict(cat_counts)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION F — PROMPT INJECTION FAILURE ANALYSIS (Top 25 FNs)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SECTION F — Prompt Injection: Top 25 False Negatives")
    print("=" * 70)

    fns_inj = sorted([r for r in inject_res if r["pred"] == 0 and r["label"] == 1],
                     key=lambda x: -x["fs"])[:25]
    print(f"\nTotal FNs: {len([r for r in inject_res if r['pred']==0 and r['label']==1])} / {len(inject_res)}")
    print(f"\n{'#':<4} {'Pat':>5} {'Sem':>5} {'Beh':>5} {'FS':>6}  Prompt (first 90 chars)")
    print("-" * 120)
    for i, r in enumerate(fns_inj, 1):
        snippet = r["prompt"][:90].replace("\n", " ")
        print(f"{i:<4} {r['ps']:>5.1f} {r['ss']:>5.1f} {r['bs']:>5.1f} "
              f"{r['fs']:>6.2f}  {snippet}")

    # Categorize injection FNs by type
    inj_fn_cats = defaultdict(int)
    for r in fns_inj:
        p = r["prompt"].lower()
        if r["bs"] >= 90 and r["ps"] == 0 and r["ss"] < 30:
            inj_fn_cats["Threshold Suppression (Beh=100, Sem Low)"] += 1
        elif "translate" in p or len(p) < 30:
            inj_fn_cats["Context Window / Short Input"] += 1
        elif any(w in p for w in ["forget what", "ignore previous", "new task"]):
            inj_fn_cats["Embedded Injection"] += 1
        elif any(c > '\u0127' for c in p):
            inj_fn_cats["Cross-Lingual Injection"] += 1
        else:
            inj_fn_cats["Instruction Smuggling (Indirect)"] += 1
    print(f"\nFN Failure Clusters: {dict(inj_fn_cats)}")

    # Root cause: how many FNs are due to threshold (fs in 30-35)?
    threshold_near = sum(1 for r in fns_inj if r["fs"] >= 30)
    low_score = sum(1 for r in fns_inj if r["fs"] < 25)
    print(f"\nFNs with fs in [30,35) (threshold suppression): {threshold_near}")
    print(f"FNs with fs < 25 (genuine low signal):           {low_score}")

    print("\n\nAudit complete.")


if __name__ == "__main__":
    main()
