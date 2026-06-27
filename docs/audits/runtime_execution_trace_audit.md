# ArgusX v11 — Runtime Execution Trace Audit

## Objective
To identify the exact, true runtime execution path used when a client submits a prompt to the production `POST /protect` API, in preparation for Phase 2B integration.

---

## 1. Complete Runtime Call Graph

Based on a rigorous trace of actual imports and object construction, the true execution graph for `POST /protect` bypasses the adaptive architecture entirely. The API is currently wired to the legacy v6 TF-IDF pipeline:

```text
POST /protect  (app/api/endpoints/protect.py)
        │
FastAPI Dependency Injection (app/api/dependencies.py)
        │  ↳ Injects `DetectionPipeline` (NOT AdaptiveDetectionPipeline)
        │
DetectionPipeline.analyze()  (app/services/detection_pipeline.py)
        │
        ├──> normalize_text()
        │
        ├──> PatternDetector.analyze()
        │
        ├──> SemanticAnalyzer.analyze()      <-- TF-IDF Engine!
        │
        ├──> BehavioralAnalyzer.analyze()
        │
        ├──> LOFDetector.analyze()
        │
ThreatScorer.compute()
        │
_enforce() decision logic
        │
ProtectResponse
```

---

## Critical Questions

### 1. Where is the semantic analyzer instantiated?
**File:** `app/services/detection_pipeline.py`
**Method:** `DetectionPipeline.__init__()`
**Code:** `self._semantic_analyzer = SemanticAnalyzer(registry.vectorizer)`
*(Note: This is the legacy TF-IDF SemanticAnalyzer. Neither SBERT nor DistilBERT is instantiated in the production API path.)*

### 2. Where is the semantic analyzer invoked?
**File:** `app/services/detection_pipeline.py`
**Method:** `DetectionPipeline.analyze()`
**Code:** `semantic_result = self._semantic_analyzer.analyze(clean_prompt)`

### 3. How many locations invoke SBERTSemanticAnalyzer?
In the production `POST /protect` execution trace: **ZERO.**
The `SBERTSemanticAnalyzer` is currently dormant code. It is only instantiated inside `SBERTDetectionPipeline`, `PromptInjectionRoute`, and `CyberThreatRoute`—none of which are currently wired into the FastAPI Dependency Injection.

### 4. Can DistilBERT replace SBERT by modifying a single location?
**No.** Replacing `SBERTSemanticAnalyzer` inside `PromptInjectionRoute` is structurally necessary, but it will have absolutely zero effect on the production API. Because `app/api/dependencies.py` currently injects `DetectionPipeline`, the entire `ThreatRouter` and Route architecture is bypassed.

### 5. Would replacing SBERT there affect...
If we replace SBERT inside the dormant `PromptInjectionRoute` (and subsequently wire it into production):
*   **Behavioral RF:** Not affected (decoupled).
*   **LOF:** Not affected.
*   **ThreatScorer:** Not affected algorithmically, but it will begin consuming polarized DistilBERT probabilities instead of SBERT/TF-IDF distances.
*   **Decision Engine & Output Scrutiny:** Not affected.

### 6. Would API responses change?
**Yes.** If we properly wire the API to use the new architecture, the `ProtectResponse.analysis` dictionary will structurally change. It will begin returning DistilBERT scores, and it will include the `selected_route` and `route_metadata` fields introduced by the `ThreatRouter` in v9.

### 7. Minimum Code Modification for Phase 2B
To successfully integrate DistilBERT into the *live production runtime*, we must bridge the gap between the API and the Adaptive architecture.
*   **Modifications required:** 2 files, 2 methods, ~4 lines of code.
    1.  **`app/routing/prompt_injection_route.py` (`__init__`)**: Swap `SBERTSemanticAnalyzer` for `DistilBERTSemanticAnalyzer`.
    2.  **`app/api/dependencies.py` (`get_pipeline`)**: Change the injected class from `DetectionPipeline(registry)` to `AdaptiveDetectionPipeline(registry)`.

### 8. Risk Assessment
*   **Regression Risk: HIGH.** We are not merely swapping SBERT for DistilBERT; wiring `dependencies.py` means hot-swapping the API directly from the legacy v6 (TF-IDF) pipeline to the v11 (Adaptive + DistilBERT) pipeline in one move.
*   **Architectural Risk: LOW.** The `AdaptiveDetectionPipeline` was explicitly engineered as a 1:1 drop-in replacement for the FastAPI endpoints.
*   **Integration Risk: MEDIUM.** The shift from TF-IDF distances to DistilBERT probabilities may interact unexpectedly with the `ThreatScorer`'s Strategy D dynamic thresholds on live traffic.
*   **Runtime Risk: HIGH.** The "device mismatch" risk identified in the Phase 1 Code Review will become a live threat. As live traffic hits DistilBERT, any tensor allocation errors will trigger 500 Internal Server Errors.

---

## 9. Phase 2B Implementation Plan

1.  **File:** `app/routing/prompt_injection_route.py`
    *   **Action:** Update the import to `from app.detection.distilbert_semantic_analyzer import DistilBERTSemanticAnalyzer`.
    *   **Action:** Inside `__init__`, set `self._semantic_analyzer = DistilBERTSemanticAnalyzer()`.
2.  **File:** `app/api/dependencies.py`
    *   **Action:** Update the import to `from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline`.
    *   **Action:** Inside `get_pipeline()`, instantiate `request.app.state._pipeline = AdaptiveDetectionPipeline(registry)`.
3.  **Verification Strategy:**
    *   Restart the FastAPI application to clear cached `app.state._pipeline`.
    *   Submit a `POST /protect` request.
    *   Verify that `analysis.selected_route` exists in the response (confirming Adaptive pipeline is active).
    *   Verify that `analysis.semantic_top_match` reads `"DistilBERT PI Prediction"` (confirming DistilBERT is active).

---

## Final Recommendation

**Option B**
Multiple invocation points (and architectural gaps) exist. The production API is currently bypassing the Threat Router entirely. Phase 2B MUST include an update to `app/api/dependencies.py` to activate the `AdaptiveDetectionPipeline`; otherwise, the DistilBERT integration will remain strictly theoretical and completely absent from production execution.
