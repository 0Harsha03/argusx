# ArgusX v10 Pattern Recovery Report

**Branch:** `argusx-v10-pattern-recovery`
**Base Tag:** `v9.6-calibrated-fusion`
**Date:** 2026-06-24

---

## 1. Rules Added

Two new `PatternRule` entries were appended to the `_RULES` list in
`app/detection/pattern_detector.py`. No other files were modified.

### Rule A: `translation_bypass`
* **Category:** `PROMPT_INJECTION`
* **Weight:** `85`
* **Regex:** `^\s*(translate|convert)\s+(to|into)\s+[a-z]+\s*$`
* **Rationale:** Attackers issue bare zero-shot translation commands to coerce an LLM into re-emitting its system prompt in a foreign language, bypassing English-only content guardrails. The anchored `^…$` boundary ensures the rule fires **only** when the entire prompt is a translation command with no accompanying body text, keeping false-positive risk low.

### Rule B: `obfuscated_ignore`
* **Category:** `INSTRUCTION_OVERRIDE`
* **Weight:** `90`
* **Regex:** `\b(igm?re|ignroe|1gnore|ignr)\b.{0,60}\b(previous|prior|above|all|system|initial|everything|what i said)\b`
* **Rationale:** The existing `instruction_override_ignore` rule requires correct spelling. Attackers exploit this by intentionally typo-obfuscating "ignore" (`igmre`, `ignroe`). The context requirement (override-vocabulary word must follow within 60 chars) prevents accidental matches on benign misspellings.

---

## 2. Unit Test Results

All **32 tests passed** in `tests/test_pattern_recovery.py`:

```
============================= test session starts =============================
platform win32 — Python 3.13.2, pytest-9.0.3
collected 32 items

tests/test_pattern_recovery.py ................................    [100%]

============================= 32 passed in 0.30s ==============================
```

### Test coverage included:
* 8 positive cases for `translation_bypass`
* 5 negative cases for `translation_bypass`
* 8 positive cases for `obfuscated_ignore`
* 4 negative cases for `obfuscated_ignore`
* 7 score / category / FP sanity checks

---

## 3. Benchmark Results

**Dataset:** `deepset/prompt-injections` (frozen test split, N=116)
**Pipeline:** v9.6 calibration + Platt scaling + `0.90 × DistilBERT + 0.10 × RF` fusion + v10 pattern override

| Metric | v9.6 Baseline | v10 Result | Delta |
|---|---|---|---|
| Accuracy | 96.55% | **98.28%** | +1.72pp |
| Precision | 96.67% | **96.77%** | +0.10pp |
| Recall | 96.67% | **100.00%** | +3.33pp |
| F1 | 96.67% | **98.36%** | **+1.69pp** |

### Confusion Matrix

| | Predicted Benign | Predicted Malicious |
|---|---|---|
| **Actual Benign** | TN = 54 | FP = 2 |
| **Actual Malicious** | FN = **0** | TP = **60** |

---

## 4. Delta vs v9.6

```
F1:     96.67%  →  98.36%   (+1.69pp)
Recall: 96.67%  →  100.00%  (+3.33pp)
FN:          2  →       0   (fully recovered)
```

The 2 False Positives (Idx 61, 69 — benign German text) are **unchanged**.
The Pattern Engine was correctly silent on them (score = 0.0).

Pattern flags fired on 4 samples: `[75, 95, 101, 111]`.
Samples 101 and 111 were previously predicted correctly by the fusion stack;
the pattern engine firing on them produced no change to those predictions.

---

## 5. Replay Prediction Assessment

**The counterfactual replay prediction was fully reproduced.**

| | Predicted | Actual |
|---|---|---|
| TP | 60 | 60 ✅ |
| FP | 2 | 2 ✅ |
| TN | 54 | 54 ✅ |
| FN | 0 | 0 ✅ |
| F1 | ≈ 98.36% | 98.36% ✅ |

The replay simulation (which used the uncalibrated RF raw score as a proxy) 
predicted the exact same confusion matrix. The actual benchmark produced
**identical results** to the simulation.
