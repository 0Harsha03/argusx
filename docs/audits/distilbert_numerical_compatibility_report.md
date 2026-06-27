# ArgusX v11 — Phase 2A Numerical Compatibility Report

## Objective
To review the numerical and mathematical compatibility between the existing `SBERTSemanticAnalyzer` and the newly implemented `DistilBERTSemanticAnalyzer` specifically concerning their interaction with the downstream `ThreatScorer`.

---

## 1. How does SBERTSemanticAnalyzer compute its final `semantic_score`?
The SBERT analyzer encodes the prompt and calculates the cosine similarity against a matrix of adversarial anchor embeddings.
1. It computes `max_sim` = the maximum raw cosine similarity value (theoretically `[-1.0, 1.0]`, practically `[0.05, 1.0]`).
2. It scales this distance metric into a `0–100` score using a linear multiplier capped at 100:
   ```python
   score = min(max_sim * 125.0, 100.0)
   ```

## 2. What numerical range does SBERT actually emit during inference?
Because SBERT embeddings are highly dense, true orthogonality (`max_sim = 0`) is rare.
*   **Benign Prompts:** Typically yield `max_sim` in the range of `0.10` to `0.20`, resulting in an emitted `score` of **12.5 to 25.0**.
*   **Borderline/Suspicious:** `max_sim` around `0.28`, resulting in a `score` of **35.0**.
*   **Highly Malicious:** `max_sim >= 0.80`, capping the `score` at **100.0**.

## 3. Is DistilBERT probability × 100 mathematically compatible with that score?
**Structurally: YES. Distributionally: NO.**

*   **Structural Compatibility:** The `ThreatScorer` simply expects a float bounded between `[0, 100]`. DistilBERT's `p * 100.0` (where `p` is the softmax probability of class 1) perfectly satisfies the `[0, 100]` contract. A probability of `1.0` maps to `100.0`, contributing the maximum `25.0` points to the `final_score` (`100.0 * 0.25`).
*   **Distributional Incompatibility:** SBERT calculates a *distance metric*, while DistilBERT calculates a *confidence probability*. Neural network classifiers (especially fine-tuned BERT variants) tend to be highly polarized. 
    *   A benign prompt in DistilBERT will likely yield `p < 0.01`, resulting in a `score` near **0.0** (much lower than SBERT's baseline of ~20.0).
    *   A malicious prompt will likely yield `p > 0.99`, resulting in a `score` near **100.0**.

## 4. Will the existing ThreatScorer thresholds (including Strategy D) behave identically?
**No, they will behave differently, though safely.**

In `ThreatScorer`, the Strategy D threshold for semantic signal agreement is:
`STRATEGY_D_SEM_SIGNAL_MIN = 35.0`

*   **Under SBERT:** This requires `max_sim >= 0.28`. Because SBERT naturally emits higher baseline scores (~20), an increase to `35.0` represents only a mild elevation in semantic proximity.
*   **Under DistilBERT:** This requires `p >= 0.35` (35% probability of Prompt Injection). Because DistilBERT benign scores sit near `0.0`, reaching `35.0` requires the model to actively register suspicion. 

**Consequence:** DistilBERT will trigger Strategy D (`semantic_score >= 35.0`) less frequently on purely benign text than SBERT, reducing False Positives (FPs) caused by Strategy D's threshold lowering. Conversely, when DistilBERT is highly confident (`score = 99.9`), the `ThreatScorer` will easily block the prompt, just as it did with SBERT.

## 5. What normalization should be applied?
**None is required to maintain architectural integrity.**

Although the internal distributions differ (Distance vs. Probability), `p * 100.0` is the most mathematically sound mapping of a binary classifier into the ArgusX `[0, 100]` scoring engine without recalibrating the downstream `ThreatScorer`. 

If strict distributional parity is ever desired in the future, DistilBERT's raw logits would require **Platt Scaling (Temperature Calibration)** before the softmax layer to soften its polarized confidence curve to match SBERT's linear distance curve. However, for Phase 2, the current linear mapping `min(p * 100.0, 100.0)` strictly satisfies the interface contract and safely interoperates with Strategy D.
