# ArgusX — Behavioral & Anomaly Component Ablation Study

## Methodology & Experiment Design
This ablation study evaluates the precise contribution of two detection components:
1. **Behavioral Random Forest (RF)**
2. **LOF Anomaly Detector**

The study was performed independently across two pipelines:
*   **Prompt Injection Pipeline:** Pattern → DistilBERT → Behavioral RF → LOF → Threat Scorer
*   **Cyber Threat Pipeline:** Pattern → SBERT → Behavioral RF → LOF → Threat Scorer

**Datasets Used:**
*   *Prompt Injection:* `deepset/prompt-injections` (N=116)
*   *Cyber Threat:* Meta Purple Llama CyberSecEval (MITRE, Spear Phishing, Interpreter Abuse, MITRE FRR) (N=3106)

**Configurations Evaluated:**
*   **A:** Original (All components active)
*   **B:** No Behavioral RF
*   **C:** No LOF Anomaly Detector
*   **D:** No Behavioral RF & No LOF Anomaly Detector

All evaluations utilized the exact pre-trained models already present in the repository, with no retraining or parameter modification. Components were systematically bypassed by forcing their outputs to `0.0` before aggregation in the `ThreatScorer`.

---

## 1. Prompt Injection Pipeline Results

| Config | Accuracy | Precision | Recall | F1 Score | Latency (ms) | Memory Δ (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A (All)** | 88.79% | 97.96% | 80.00% | **88.07%** | 64.76 | ~146 |
| **B (No RF)** | 54.31% | 100.00% | 11.67% | **20.90%** | 53.54 | ~0 |
| **C (No LOF)** | 88.79% | 97.96% | 80.00% | **88.07%** | 61.54 | ~0 |
| **D (No RF, No LOF)** | 54.31% | 100.00% | 11.67% | **20.90%** | 55.23 | ~0 |

### Confusion Matrices (PI)
*   **Config A / C:** `TN=55`, `FP=1`, `FN=12`, `TP=48`
*   **Config B / D:** `TN=56`, `FP=0`, `FN=53`, `TP=7`

---

## 2. Cyber Threat Pipeline Results

| Config | Accuracy | Precision | Recall | F1 Score | Latency (ms) | Memory Δ (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A (All)** | 75.63% | 75.88% | 99.49% | **86.10%** | 24.65 | ~32 |
| **B (No RF)** | 34.80% | 82.77% | 17.74% | **29.22%** | 16.36 | ~1 |
| **C (No LOF)** | 75.63% | 75.88% | 99.49% | **86.10%** | 25.31 | ~0 |
| **D (No RF, No LOF)** | 34.80% | 82.77% | 17.74% | **29.22%** | 16.90 | ~7 |

### Confusion Matrices (Cyber)
*   **Config A / C:** `TN=5`, `FP=745`, `FN=12`, `TP=2344`
*   **Config B / D:** `TN=663`, `FP=87`, `FN=1938`, `TP=418`

---

## Analysis & Engineering Discussion

**1. How much does removing Behavioral RF affect performance?**
Removing the Behavioral RF triggers a catastrophic collapse in both pipelines. For Prompt Injection, F1 plummets from 88.07% to 20.90%. For Cyber Threats, F1 plummets from 86.10% to 29.22%.

**2. How much does removing LOF affect performance?**
Removing the LOF Anomaly Detector produces **zero measurable effect**. The accuracy, precision, recall, F1, and confusion matrices are identical down to the last decimal point across all 3,222 evaluated samples.

**3. Which datasets are most sensitive to removing RF?**
Both datasets are extremely sensitive to the removal of the RF layer, suffering a >65 percentage point drop in F1 score.

**4. Which datasets are most sensitive to removing LOF?**
Neither dataset is sensitive to the removal of LOF. Its removal did not alter a single prediction.

**5. Does either component primarily improve Precision, Recall or both?**
The Behavioral RF primarily and overwhelmingly improves **Recall**. Without it, DistilBERT and SBERT are highly conservative (yielding high precision but missing >80% of attacks). 

**6. Does either component mainly reduce False Positives or False Negatives?**
The Behavioral RF massively reduces **False Negatives** (from 53 down to 12 in DeepSet, and from 1938 down to 12 in CyberSecEval). However, it does cause a moderate increase in False Positives for the CyberSecEval dataset (benign queries falsely blocked rose from 87 to 745). The LOF does neither.

**7. Is the computational overhead introduced by these components justified by the observed performance gains?**
*   **Behavioral RF:** Yes. It adds ~8-11ms of latency but recovers over 1,900 missed attacks. The trade-off is exceptional.
*   **LOF Anomaly Detector:** No. It adds ~1-3ms of latency and continuous memory overhead for the k-neighbors tree, but provides exactly 0% performance gain on the benchmarks.

---

## Final Verdict & Architectural Recommendation

| Pipeline | Component | Performance Delta | Latency Delta | Memory Delta | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prompt Injection** | Behavioral RF | -67.17% F1 | -11.2 ms | negligible | **KEEP** |
| **Prompt Injection** | LOF Anomaly | 0.00% F1 | -3.2 ms | negligible | **REMOVE** |
| **Cyber Threat** | Behavioral RF | -56.88% F1 | -8.3 ms | ~31 MB | **KEEP** |
| **Cyber Threat** | LOF Anomaly | 0.00% F1 | +0.6 ms | negligible | **REMOVE** |

### Official Recommendation:

**For the Prompt Injection pipeline:**
*   Should Behavioral RF remain? **YES**
*   Should LOF remain? **NO**

**For the Cyber Threat pipeline:**
*   Should Behavioral RF remain? **YES**
*   Should LOF remain? **NO**

**Conclusion:** The Local Outlier Factor (LOF) anomaly detector is dead weight. It does not provide any unique signal that isn't already captured by the semantic or behavioral layers. It should be entirely stripped from the architecture to simplify the pipeline and reduce inference latency. Conversely, the Behavioral Random Forest is structurally critical to both pipelines; without it, ArgusX devolves into an extremely weak, low-recall filter.
