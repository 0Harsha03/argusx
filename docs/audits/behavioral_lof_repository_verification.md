# ArgusX — Behavioral & LOF Repository Verification Audit

## Objective
To determine whether the Behavioral Random Forest (RF) and Local Outlier Factor (LOF) models are structurally shared between the Prompt Injection and Cyber Threat pipelines, or if they represent independent instances/artifacts.

---

## 1. Physical Artifact Inventory
Inspection of the `app/models/artifacts/` directory reveals exactly three `.pkl` files physically present in the repository:
1. `anomaly_detector.pkl` (9.03 MB)
2. `behavioral_model.pkl` (508 KB)
3. `vectorizer.pkl` (181 KB)

**Conclusion:** There is only **ONE** physical Behavioral RF artifact and **ONE** LOF artifact. There are no separate models for Prompt Injection and Cyber Threats.

---

## A. Behavioral RF Verification

**Verdict:** **SHARED**

**Evidence from Repository:**
1. **Model Loading:** `app/services/model_registry.py` defines a singleton registry. During initialization, it calls `self._load(settings.BEHAVIORAL_MODEL_PATH)` exactly once to load `behavioral_model.pkl` into memory. 
2. **Pipeline Injection:** 
   * `app/routing/prompt_injection_route.py` initializes its `BehavioralAnalyzer` by passing `behavioral_model` and `vectorizer` from the registry.
   * `app/routing/cyber_threat_route.py` initializes its `BehavioralAnalyzer` by passing the *exact same* `behavioral_model` and `vectorizer` objects from the same registry.
3. **Training Scripts:** A global repository search for `joblib.dump` and `.dump(` yields **zero results**. There are no scripts that dynamically train or split the Behavioral RF. The single artifact was pre-trained offline on a unified dataset (CVEs + Synthetic PI) and committed as a static blob.

---

## B. LOF Verification

**Verdict:** **SHARED**

**Evidence from Repository:**
1. **Model Loading:** `app/services/model_registry.py` loads `anomaly_detector.pkl` exactly once.
2. **Pipeline Injection:** Both `PromptInjectionRoute` and `CyberThreatRoute` instantiate the `AnomalyDetector` using the singular `anomaly_detector` object injected from the registry.

---

## 6. ACTUAL Runtime Architecture Diagram
Based on `app/services/adaptive_detection_pipeline.py` and `app/routing/threat_router.py`, the true runtime execution graph inside the application is unified at the ends but bifurcated semantically in the middle. 

Because both Route wrappers use SBERT and instantiate their downstream layers identically, the *actual* execution path invoked by the router currently looks like this:

```text
                            [ raw_prompt ]
                                  │
                                  ▼
                          [ Preprocessor ]
                                  │
                                  ▼
                   [ PatternDetector (Regex engine) ]
                                  │
                                  ▼
                           [ ThreatRouter ]
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      (PromptInjectionRoute)               (CyberThreatRoute)
                 │                                 │
           [ SBERT Engine ]                  [ SBERT Engine ]
                 │                                 │
    [ Behavioral RF (behavioral_model.pkl) ]  [ Behavioral RF (behavioral_model.pkl) ]
                 │                                 │
    [ LOF Anomaly (anomaly_detector.pkl)   ]  [ LOF Anomaly (anomaly_detector.pkl)   ]
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                  [ ThreatScorer (Aggregation) ]
                                  │
                                  ▼
                [ Decision Engine (BLOCK/ALLOW) ]
```

*Note: While conceptually the pipelines are split by the router, they both physically point to the exact same pre-loaded artifacts in memory for Semantic, Behavioral, and Anomaly analysis.*
