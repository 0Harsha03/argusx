# ArgusX Architectural Audit Report

This report was generated purely by inspecting the repository's source code, avoiding any assumptions based on filenames alone.

---

## Part 1 — Repository Tree

```text
C:\Users\shars\OneDrive\Desktop\argusx
├── main.py                                      # Core production code (FastAPI Entry Point)
├── app/
│   ├── __init__.py                              # Utility
│   ├── api/
│   │   ├── dependencies.py                      # Configuration / Dependency injection (get_pipeline)
│   │   ├── router.py                            # Configuration / FastApi routing setup
│   │   ├── schemas.py                           # Core production code / Pydantic models
│   │   ├── __init__.py                          # Utility
│   │   └── endpoints/
│   │       ├── analyze.py                       # Core production code / /analyze endpoint
│   │       ├── health.py                        # Utility / /health endpoint
│   │       ├── logs.py                          # Utility / /logs endpoint
│   │       ├── protect.py                       # Core production code / /protect endpoint
│   │       └── __init__.py                      # Utility
│   ├── core/
│   │   ├── config.py                            # Configuration / App Settings
│   │   ├── database.py                          # Core production code / DB Engine setup
│   │   ├── logging_config.py                    # Configuration / Logging config
│   │   └── __init__.py                          # Utility
│   ├── detection/
│   │   ├── anomaly_detector.py                  # Core production code / Anomaly LOF Layer
│   │   ├── behavioral_analyzer.py               # Core production code / RF Classifier Layer
│   │   ├── pattern_detector.py                  # Core production code / Regex Engine Layer
│   │   ├── semantic_analyzer.py                 # Core production code / TF-IDF Layer
│   │   ├── sbert_semantic_analyzer.py           # Experimental / Unused SBERT layer alternative
│   │   ├── threat_scorer.py                     # Core production code / Decision Engine
│   │   └── __init__.py                          # Utility
│   ├── models/
│   │   ├── db_models.py                         # Core production code / SQLAlchemy Schemas
│   │   ├── __init__.py                          # Utility
│   │   └── artifacts/
│   │       ├── anomaly_detector.pkl             # Core production code / LOF Model artifact
│   │       ├── behavioral_model.pkl             # Core production code / RF Model artifact
│   │       └── vectorizer.pkl                   # Core production code / TF-IDF Vectorizer artifact
│   ├── output_scrutiny/
│   │   ├── scrutinizer.py                       # Core production code / Output filter layer
│   │   └── __init__.py                          # Utility
│   ├── routing/
│   │   ├── base_route.py                        # Experimental / Base logic for Threat Router
│   │   ├── cyber_threat_route.py                # Experimental / Cyber threat isolated route
│   │   ├── prompt_injection_route.py            # Experimental / PI isolated route
│   │   ├── threat_router.py                     # Experimental / Multi-route logic
│   │   └── __init__.py                          # Utility
│   ├── services/
│   │   ├── adaptive_detection_pipeline.py       # Experimental / Unused adaptive pipeline
│   │   ├── detection_pipeline.py                # Core production code / Linear 4-layer Orchestrator
│   │   ├── llm_service.py                       # Core production code / LLM interaction abstraction
│   │   ├── model_registry.py                    # Core production code / Singleton for loading .pkl models
│   │   ├── sbert_detection_pipeline.py          # Experimental / Unused pipeline running SBERT
│   │   └── __init__.py                          # Utility
│   └── utils/
│       ├── preprocessor.py                      # Core production code / Text normalization
│       └── __init__.py                          # Utility
├── data/
│   ├── anchors_v7_baseline.json                 # Dataset / Anchors for SBERT
│   ├── pi_augmentation/                         # Dataset / Synthetic datasets
│   └── pi_corpus/                               # Dataset / Training data for ML components
├── docker/                                      # Configuration / Deployment artifacts
├── evaluation/                                  # Evaluation / Testing and metric generation scripts
├── models/
│   ├── distilbert_pi/                           # Research / HF Transformer artifact
│   ├── distilbert_pi_augmented/                 # Research / HF Transformer artifact
│   └── distilbert_sst2_pi/                      # Research / HF Transformer artifact
├── results/                                     # Documentation / Test outputs and experiment results
├── scripts/                                     # Scripts / Offline benchmarking, evaluation, and training
│   ├── analyze_residual_errors.py               # Evaluation script
│   ├── audit_cyberseceval.py                    # Evaluation script
│   ├── audit_rf_for_fusion.py                   # Evaluation script
│   ├── build_pi_corpus.py                       # Training script
│   ├── eval_blind_benchmark.py                  # Evaluation script
│   ├── eval_calibrated_fusion.py                # Evaluation script
│   ├── eval_context_dilution.py                 # Evaluation script
│   ├── eval_cyberseceval.py                     # Evaluation script
│   ├── eval_deepset.py                          # Evaluation script
│   ├── eval_distilbert_deepset.py               # Evaluation script
│   ├── eval_distilbert_rf_fusion.py             # Evaluation script
│   ├── eval_jailbreakbench.py                   # Evaluation script
│   ├── eval_v10_pattern_recovery.py             # Evaluation script
│   ├── eval_v7_sbert.py                         # Evaluation script
│   ├── forensics_fusion_discrepancy.py          # Evaluation script
│   ├── threshold_study.py                       # Evaluation script
│   ├── train_distilbert_pi.py                   # Training script
│   └── validate_v9_identity.py                  # Evaluation script
└── tests/                                       # Tests / Pytest definitions
```

---

## Part 2 — Component Classification

### API Layer
* **Responsibility:** Request handling, validation, response structuring.
* **Files:** `app/api/router.py`, `app/api/endpoints/*.py`, `app/api/schemas.py`, `app/api/dependencies.py`
* **Dependencies:** `fastapi`, `pydantic`, `app.services.detection_pipeline`

### Detection Engines
* **Responsibility:** Executing specific detection algorithms on prompt text.
* **Files:** `app/detection/pattern_detector.py`, `app/detection/semantic_analyzer.py`, `app/detection/behavioral_analyzer.py`, `app/detection/anomaly_detector.py`, `app/detection/threat_scorer.py`
* **Dependencies:** `sklearn`, `numpy`, `re`

### Services
* **Responsibility:** Core application logic, model loading, orchestrating detection layers, LLM I/O.
* **Files:** `app/services/detection_pipeline.py`, `app/services/model_registry.py`, `app/services/llm_service.py`
* **Dependencies:** `app.detection.*`, `joblib`

### Experimental Routing (Unused in Production)
* **Responsibility:** Attempting to split detection logic into task-specific routes based on threat class.
* **Files:** `app/routing/*.py`, `app/services/adaptive_detection_pipeline.py`
* **Dependencies:** `app.detection.*`

### Models (Database & Artifacts)
* **Responsibility:** SQLAlchemy database definitions and pickled inference models.
* **Files:** `app/models/db_models.py`, `app/models/artifacts/*.pkl`
* **Dependencies:** `sqlalchemy`

### Scripts & Evaluation
* **Responsibility:** Offline benchmarking, data processing, model training, and forensics.
* **Files:** `scripts/*.py`, `evaluation/*.py`
* **Dependencies:** `transformers`, `datasets`, `pandas`

---

## Part 3 — Dependency Graph (Active Production Route)

```text
main.py
 │
 ├── app.core.database (Initializes SQLite)
 ├── app.services.model_registry (Loads .pkl artifacts)
 │
 └── app.api.router
      │
      └── app.api.endpoints.protect
           │
           ├── app.services.llm_service (Handles downstream LLM I/O)
           ├── app.output_scrutiny.scrutinizer (Validates output)
           │
           └── app.services.detection_pipeline (Orchestrator)
                │
                ├── app.utils.preprocessor (Text Normalization)
                ├── app.detection.pattern_detector (Regex Rules)
                ├── app.detection.semantic_analyzer (TF-IDF Cosine Similarity)
                ├── app.detection.behavioral_analyzer (Random Forest)
                ├── app.detection.anomaly_detector (Local Outlier Factor)
                └── app.detection.threat_scorer (Aggregates logic & thresholds)
```

*(Note: `app.routing` and `sbert` are conspicuously absent from this graph as they are completely unreferenced by the active request path).*

---

## Part 4 — Runtime Execution Flow (`POST /protect`)

1. **Entry Point:** The HTTP POST request hits `app/api/endpoints/protect.py`.
2. **Dependencies:** FastAPI resolves the injected `pipeline: DetectionPipeline` via `app.api.dependencies.get_pipeline()`.
3. **Execution start:** `protect.py` extracts the raw prompt, `session_id`, and IP.
4. **Orchestrator Call:** `pipeline.analyze()` is called on the `DetectionPipeline` instance.
5. **Layer 0 (Preprocessor):** `normalize_text()` truncates and prepares the string.
6. **Layer 1 (Pattern):** `PatternDetector.analyze()` runs synchronously using regex rules.
7. **Layer 2 (Semantic):** `SemanticAnalyzer.analyze()` runs synchronously using a TF-IDF vectorizer to calculate cosine similarity.
8. **Layer 3 (Behavioral):** `BehavioralAnalyzer.analyze()` runs synchronously using the RF classifier.
9. **Layer 4 (Anomaly):** `AnomalyDetector.analyze()` runs synchronously using LOF.
10. **Decision Aggregation:** `ThreatScorer.compute()` aggregates the outputs of all 4 layers using defined weights and returns a `ThreatScore` object with a decision (`ALLOW`, `FLAG`, `SANITIZE`, `BLOCK`).
11. **Enforcement:** `protect.py` processes the decision via the `_enforce` helper. If `BLOCK`, it returns immediately. If `SANITIZE` or `ALLOW`, it optionally applies redacted text and delegates to `_llm_service.generate()`.
12. **Output Scrutiny:** The LLM's response is passed to `_output_scrutinizer.scrutinize()`.
13. **Audit:** The result is saved to the local database via `db.add(AnalysisLog(...))`.
14. **Final Response:** The endpoint returns the structured `ProtectResponse`.

---

## Part 5 — Detection Architecture

### Pattern Detector
* **File:** `app/detection/pattern_detector.py`
* **Class:** `PatternDetector`
* **Purpose:** Regex-based heuristic detection.
* **Used In:** Production (`DetectionPipeline`)
* **Status:** **Active in Production**

### TF-IDF Semantic Analyzer
* **File:** `app/detection/semantic_analyzer.py`
* **Class:** `SemanticAnalyzer`
* **Purpose:** Calculates TF-IDF cosine similarity against anchor prompts.
* **Used In:** Production (`DetectionPipeline`)
* **Status:** **Active in Production**

### SBERT Semantic Analyzer
* **File:** `app/detection/sbert_semantic_analyzer.py`
* **Class:** `SBERTSemanticAnalyzer`
* **Purpose:** Experimental transformer-based semantic analysis.
* **Used In:** `sbert_detection_pipeline.py` (which is dead code) and `eval_v7_sbert.py`.
* **Status:** **Research / Unused in Production**

### DistilBERT Detector
* **File:** Offline in `models/distilbert_pi/`
* **Purpose:** HuggingFace classifier model meant for deep pipeline fusion.
* **Used In:** Evaluation scripts in `scripts/` (e.g. `eval_distilbert_rf_fusion.py`).
* **Status:** **Research / Evaluation Only. Never executed in `main.py`.**

### Behavioral RF
* **File:** `app/detection/behavioral_analyzer.py`
* **Class:** `BehavioralAnalyzer`
* **Purpose:** Random Forest classifier utilizing TF-IDF n-grams + heuristic metadata.
* **Used In:** Production (`DetectionPipeline`)
* **Status:** **Active in Production**

### Anomaly Detector
* **File:** `app/detection/anomaly_detector.py`
* **Class:** `AnomalyDetector`
* **Purpose:** Local Outlier Factor calculation.
* **Used In:** Production (`DetectionPipeline`)
* **Status:** **Active in Production**

### Decision Engine
* **File:** `app/detection/threat_scorer.py`
* **Class:** `ThreatScorer`
* **Purpose:** Weighted aggregation of all active layers into a final Threat Score.
* **Used In:** Production (`DetectionPipeline`)
* **Status:** **Active in Production**

---

## Part 6 — Architecture Diagram

```text
       User Request
            │
            ▼
        FastAPI App (main.py)
            │
            ▼
     POST /api/v1/protect
            │
            ├───> [DetectionPipeline]
            │          │
            │          ├── Layer 1: PatternDetector (Regex)
            │          ├── Layer 2: SemanticAnalyzer (TF-IDF Cosine Sim)
            │          ├── Layer 3: BehavioralAnalyzer (Random Forest)
            │          └── Layer 4: AnomalyDetector (LOF)
            │          │
            │          └──> ThreatScorer (Aggregator -> ALLOW/FLAG/SANITIZE/BLOCK)
            │
            ▼
    [Enforcement Filter] ───── (If BLOCK) ────────> Reject Request
            │
      (If ALLOW/SANITIZE)
            │
            ▼
       LLMService ──────> Generates Response
            │
            ▼
    OutputScrutinizer ──> Applies response security checks
            │
            ▼
     ProtectResponse
```

---

## Part 7 — Version Mapping

By tracking code history and the current state, we can map:
* **v1.0 Implementation:** `DetectionPipeline`, `ThreatScorer`, `PatternDetector`, `SemanticAnalyzer`, `BehavioralAnalyzer`, `AnomalyDetector`.
* **v2.0 Implementation:** `LLMService`, `protect.py` endpoint, `output_scrutiny`.
* **v7 Implementation:** `sbert_semantic_analyzer.py` (Never successfully merged into production routing).
* **v8 Implementation:** `app/routing/*` (Adaptive / Threat Router architecture - Never executed locally).
* **v9 Implementation:** `scripts/eval_calibrated_fusion.py` (DistilBERT benchmark models - Evaluated offline but never deployed).
* **v10 Implementation:** Addition of `translation_bypass` and `obfuscated_ignore` regex rules into `pattern_detector.py`.

**Currently Executed Local Architecture:** The application natively runs the **v2.0 Middleware Architecture** (linear execution of the v1.0 pipeline, wrapping the `LLMService`), heavily utilizing the `v10` pattern rule updates.

---

## Part 8 — Unused or Dead Code

Based on explicit import traces from the entry point (`main.py`), the following are dead/experimental code and models:

1. **`app/routing/*` (Adaptive Routing Engine)**
   * Not imported anywhere except `adaptive_detection_pipeline.py`.
   * **Reason:** It was developed as an experimental upgrade but the API (`dependencies.py`) still hard-codes injection of `DetectionPipeline`.
2. **`app/services/adaptive_detection_pipeline.py`**
   * Never imported by `main.py` or API routers.
3. **`app/services/sbert_detection_pipeline.py`**
   * Never imported.
4. **`app/detection/sbert_semantic_analyzer.py`**
   * Only imported by dead/offline scripts.
5. **`models/distilbert_pi/*`**
   * HuggingFace checkpoints. Loaded extensively in offline benchmark scripts located in `/scripts`, but `ModelRegistry` only mounts `.pkl` sklearn models. DistilBERT does not exist in the production runtime.

---

## Part 9 — Final Summary

1. **Production Architecture:** A rigid, linear 4-layer security pipeline (Pattern, TF-IDF Semantic, Behavioral RF, Anomaly LOF) managed by `DetectionPipeline` and aggregated by `ThreatScorer`.
2. **Research Architecture:** A graveyard of parallel capabilities (SBERT, DistilBERT, Adaptive Threat Routing) that were built and evaluated extensively in `/scripts`, but never permanently wired into the live `main.py` -> `protect.py` flow.
3. **Active Runtime Pipeline:** The `DetectionPipeline` class instantiated by `dependencies.py` is the only pipeline actively processing inputs in real-time.
4. **Entry Points:** `main.py` serves the FastAPI server, exposing `/protect` and `/analyze` as the primary interaction mechanisms.
5. **Key Services:** `DetectionPipeline` (security logic), `LLMService` (model routing), and `ModelRegistry` (in-memory `.pkl` artifact holder).
6. **Overall Component Interaction:** The architecture acts as a standard proxy firewall. It intercepts prompts, calculates an internal heuristic+ML threat score, optionally sanitizes the text, passes the surviving text to an LLM, sanitizes the outbound output, and returns the combined package to the user. All logic executes synchronously.
