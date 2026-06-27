# ArgusX v11 — Architectural Consolidation Audit

This audit evaluates the feasibility of an adaptive, multi-route architecture (v11) by inspecting the structural and mathematical properties of existing ArgusX components.

---

## PART 1 — Component Inventory

| Component | File | Class | Purpose | Inputs | Outputs | Dependencies | Usage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pattern Engine** | `app/detection/pattern_detector.py` | `PatternDetector` | Regex heuristic matching | Raw Text | `PatternDetectionResult` | `re` | Production Pipeline | Production |
| **Threat Router** | `app/routing/threat_router.py` | `ThreatRouter` | Routes prompt based on Pattern category | Text, Pattern metadata | `RoutingDecision` | `PromptInjectionRoute`, `CyberThreatRoute` | Adaptive Pipeline | Research |
| **TF-IDF Semantic Analyzer** | `app/detection/semantic_analyzer.py` | `SemanticAnalyzer` | Lexical cosine similarity | Text | `SemanticAnalysisResult` | `sklearn.feature_extraction` | Production Pipeline | Production |
| **SBERT Semantic Analyzer** | `app/detection/sbert_semantic_analyzer.py` | `SBERTSemanticAnalyzer` | Dense embedding similarity | Text | `SemanticAnalysisResult` | `sentence_transformers` | Research Pipelines | Research |
| **DistilBERT Prompt Injection** | `models/distilbert_pi/` | N/A (Offline) | Deep classification model | Text Tokens | Logits / Probs | `transformers`, `torch` | Benchmark Scripts | Research |
| **Behavioral Analyzer** | `app/detection/behavioral_analyzer.py` | `BehavioralAnalyzer` | Evaluates adversarial syntax properties | Text, Pattern metadata | `BehavioralAnalysisResult` | `vectorizer.pkl`, `behavioral_model.pkl` | Production Pipeline | Production |
| **Random Forest** | `app/models/artifacts/behavioral_model.pkl` | `RandomForestClassifier` | Machine learning classification | TF-IDF Vector | Probability (0-1) | `sklearn.ensemble` | `BehavioralAnalyzer` | Production |
| **Feature Extractor** | `app/models/artifacts/vectorizer.pkl` | `TfidfVectorizer` | Text tokenization/vectorization | Text | TF-IDF Vector | `sklearn.feature_extraction` | `BehavioralAnalyzer`, `AnomalyDetector` | Production |
| **LOF Anomaly Detector** | `app/detection/anomaly_detector.py` | `AnomalyDetector` | Unsupervised structural novelty detection | Text | `AnomalyDetectionResult` | `anomaly_detector.pkl`, `vectorizer.pkl` | Production Pipeline | Production |
| **Threat Scorer** | `app/detection/threat_scorer.py` | `ThreatScorer` | Score aggregation & thresholding | Layer 1-4 results | `ThreatScore` | Python `math` | Production Pipeline | Production |
| **Decision Engine** | `app/detection/threat_scorer.py` | Embedded inside `ThreatScorer` | Resolves final firewall action | Final Score (0-100) | ALLOW / FLAG / SANITIZE / BLOCK | Internal Thresholds | Production Pipeline | Production |
| **Output Scrutiny** | `app/output_scrutiny/scrutinizer.py` | `OutputScrutinizer` | Post-generation safety filter | LLM Output Text | `ScrutinyResult` | Regex, heuristics | `protect.py` | Production |
| **LLM Service** | `app/services/llm_service.py` | `LLMService` | Upstream model integration | Cleaned Prompt | LLM Response | External APIs | `protect.py` | Production |

---

## PART 2 — Behavioral Analyzer Audit

**Implementation Analyzed:** `BehavioralAnalyzer` in `behavioral_analyzer.py`

**Feature Extraction:**
1. **Mathematical:** A generic `TfidfVectorizer` transforms the prompt into an n-gram space, which is fed to the `RandomForestClassifier`.
2. **Heuristic Tokens:** The code explicitly checks for linguistic flags: `_OVERRIDE_KEYWORDS`, `_EXFIL_KEYWORDS`, `_CHAINING_KEYWORDS`, `_ROLE_KEYWORDS`.

**Is it specific to Prompt Injection?**
No. While `_OVERRIDE_KEYWORDS` and `_ROLE_KEYWORDS` are highly specific to PI and Jailbreaks, the model also extracts `_EXFIL_KEYWORDS` (e.g. "database", "dump", "credentials") and `_CHAINING_KEYWORDS` (e.g. "step", "phase") which are universal cyber-threat indicators. Furthermore, the RF model was trained explicitly on Exploit-DB CVE payloads alongside synthetic PI prompts.

**Can it operate on Cyber Threat prompts without modification?**
Yes. The RF model's "Class 0" and "Class 1" mapping inherently tracks both exploitation and injection syntax.

**Verdict:** **SHARE**. The feature space and extraction heuristics are fully generalized across security domains.

---

## PART 3 — Anomaly Detector Audit

**Implementation Analyzed:** `AnomalyDetector` in `anomaly_detector.py`

**Feature Vector & Dimensionality:**
The exact same `TfidfVectorizer` used by the Behavioral Analyzer maps the text into a fixed-dimensional space representing the entire known training corpus vocabulary.

**Training Assumptions & Space:**
The `LocalOutlierFactor` calculates distances (k-neighbors) relative to the training distribution. Because the vectorizer and LOF models were fitted simultaneously on the *complete* PI + Exploit corpus, PI and Cyber Threats effectively define different dense clusters within the same topological space.

**Shared vs Separate:**
Anomalous structural inputs (e.g., heavily obfuscated payloads or base64 encoding) will sit far outside the boundaries of both the PI cluster and the Cyber Threat cluster. A single global LOF space natively captures novelty against *all* known benign and malicious data. Separating them would create artificial blind spots.

**Verdict:** **SHARE**. One shared anomaly detector is mathematically correct and computationally efficient.

---

## PART 4 — Threat Scorer Audit

**Implementation Analyzed:** `ThreatScorer.compute()` in `threat_scorer.py`

**Inputs & Dependencies:**
The `compute()` function accepts four floats: `pattern_score`, `semantic_score`, `behavioral_score`, and `anomaly_score`, plus generic boolean flags. It possesses **zero knowledge** of what underlying models (DistilBERT, SBERT, TF-IDF) generated the scores. 

**Model Agnosticism:**
The algorithm utilizes standard normalization and `Strategy D` conditional thresholds. It treats a 95/100 from DistilBERT identically to a 95/100 from SBERT.

**Verdict:** **SHARE**. The scoring engine is completely abstracted from the detection mechanisms.

---

## PART 5 — Decision Engine Audit

**Implementation Analyzed:** Threshold routing at the end of `threat_scorer.py`

**Logic:**
The decision engine maps the aggregated `final_score` (0-100) to firewall actions (`BLOCK_THRESHOLD = 85`, `SANITIZE_THRESHOLD = 70`). The logic does not branch based on whether the threat is a "Jailbreak" or a "CVE Exploit"—an 86 is universally blocked.

**Verdict:** **SHARE**. One unified security policy (Decision Engine) must govern the firewall, regardless of the pipeline branch taken.

---

## PART 6 — Pattern Engine Audit

**Implementation Analyzed:** `_RULES` array in `pattern_detector.py`

**Rule Categorization:**
*   **Prompt Injection:** `translation_bypass`, `prompt_injection_delimiter`, `prompt_injection_indirect`
*   **System Prompt Extraction:** `role_manipulation_system_prompt`, `system_extraction_extended`
*   **Jailbreak:** `jailbreak_dan`, `jailbreak_pretend`, `jailbreak_hypothetical`, `jailbreak_named_persona`, `jailbreak_authorization_claim`, `jailbreak_scenario_extended`
*   **General Linguistic:** `instruction_override_ignore`, `instruction_override_new_task`, `instruction_override_extended`, `obfuscated_ignore`, `role_manipulation_persona`, `social_engineering_authority`
*   **Malware:** `malware_generation_direct`, `malware_payload_request`, `malware_evasion_technique`, `malware_generation_extended`, `malware_network_attack`
*   **Credential Theft:** `exfil_credentials`, `credential_theft_direct`, `credential_dumping_tool`, `credential_attack_extended`, `credential_infra_attack`
*   **General Cyber Threat:** `code_execution_dangerous`, `attack_chaining_step`, `exploitation_request`, `privilege_escalation`, `exploit_cve_request`, `phishing_attack`, `mass_attack_automation`, `exfil_database`

**One Engine vs Separate?**
The `re` engine evaluates patterns with negligible overhead. Combining them in one engine allows the `PatternDetector` to return a `category` tag, which the `ThreatRouter` relies on to make branching decisions downstream. Splitting the engine would destroy this early classification capability.

**Verdict:** **SHARE**. One unified pattern library is preferable.

---

## PART 7 — Semantic Layer Audit

**Implementation Analyzed:** DistilBERT vs SBERT vs TF-IDF properties and Benchmark data

*   **TF-IDF:** Lexical distance. Broad generalization but fails to understand deep semantic intent.
*   **SBERT:** Dense contextual embeddings. Excellent at cross-domain generalization and intent matching via anchor comparisons.
*   **DistilBERT (Prompt Injection Checkpoint):** Softmax sequence classifier heavily fine-tuned on the `deepset/prompt-injections` dataset. 

**Benchmark Evidence (`results/cyberseceval_generalization/metrics.json`):**
The repository's internal benchmark proves that DistilBERT achieves an **F1 of 98.36% on Prompt Injections**, but completely collapses when exposed to the `CyberSecEval` cyber-threat dataset, yielding an **F1 of 0.32** (`TN=0`, `FP=750`). 

DistilBERT is severely overfitted to Prompt Injection syntax. It fundamentally cannot generalize to Cyber Threats. SBERT, conversely, relies on distance to known anchors, making it highly robust across domains.

**Verdict:** **SEPARATE**. Using highly specialized, overfitted classification transformers requires routing Prompt Injections to DistilBERT and standard Cyber Threats to an embedding similarity model (SBERT).

---

## PART 8 — Integration Feasibility

Can we natively execute the proposed bifurcated pipeline architecture?

1.  **User Prompt**
2.  **Pattern Analysis** *(Shared)*
3.  **Threat Router** *(Routes based on Pattern Category)*
4.  **Semantic Fork:**
    *   **Route A (Prompt Injection):** DistilBERT *(Separate)*
    *   **Route B (Cyber Threat):** SBERT *(Separate)*
5.  **Behavioral RF** *(Shared)*
6.  **LOF** *(Shared)*
7.  **Threat Scorer** *(Shared)*
8.  **Decision Engine** *(Shared)*
9.  **LLM**
10. **Output Scrutiny**

**Technical Feasibility:** **HIGH**
*   **Existing Code:** `threat_router.py` already natively inspects the Pattern category and dispatches to `PromptInjectionRoute` or `CyberThreatRoute`. All post-semantic shared components (`BehavioralAnalyzer`, `AnomalyDetector`, `ThreatScorer`) are already fully abstracted and agnostic.
*   **Modifications Required:** 
    1. `app/routing/prompt_injection_route.py`: Currently delegates to SBERT; must be rewritten to invoke a new DistilBERT wrapper class.
    2. `app/services/model_registry.py`: Must be upgraded to load HuggingFace PyTorch `AutoModelForSequenceClassification` checkpoints into memory alongside `.pkl` files.
    3. `app/api/dependencies.py`: Must inject `AdaptiveDetectionPipeline` into `/protect` instead of the legacy `DetectionPipeline`.
*   **Implementation Complexity:** Moderate.

---

## PART 9 — Final Verdict

| Component | Share | Separate | Reason |
| :--- | :--- | :--- | :--- |
| **Pattern Engine** | **SHARE** | | Highly efficient regex execution; required for early multi-class Threat Routing. |
| **Threat Router** | **SHARE** | | Central dispatcher responsible for branching the semantic execution path. |
| **Semantic Model** | | **SEPARATE** | DistilBERT perfectly captures PI (98% F1) but fails on Cyber Threats (0.32 F1). Domain separation is mathematically required. |
| **Behavioral RF** | **SHARE** | | Generalizes perfectly across PI and CVE data; relies on agnostic TF-IDF syntax heuristics. |
| **Feature Extraction** | **SHARE** | | Single TF-IDF space unifies structural representation for RF and LOF layers. |
| **LOF Anomaly** | **SHARE** | | A single cluster-distance calculation reliably identifies novelty against all known attack vectors. |
| **Threat Scorer** | **SHARE** | | Purely numerical aggregator; completely blind to upstream semantic implementations. |
| **Decision Engine**| **SHARE** | | Final firewall policy (ALLOW/BLOCK) must be uniform regardless of attack type. |
| **Output Scrutiny**| **SHARE** | | Secures the LLM's response output regardless of what triggered the generation. |
| **LLM Service** | **SHARE** | | Stateless upstream I/O abstraction. |

---

## PART 10 — Final Recommended ArgusX v11 Architecture

```text
                                  [ User Request ]
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Pattern Engine (1)    │  <-- Executes 35+ unified regex rules
                            └─────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │     Threat Router       │  <-- Branches based on pattern category
                            └─────────────────────────┘
                                     │         │
                   ┌─────────────────┘         └─────────────────┐
    (Prompt Injection Route)                              (Cyber Threat Route)
                   ▼                                             ▼
      ┌────────────────────────┐                    ┌────────────────────────┐
      │ Semantic Engine (2A)   │                    │ Semantic Engine (2B)   │
      │       DistilBERT       │                    │         SBERT          │
      │  (Deep classification) │                    │  (Cosine Similarity)   │
      └────────────────────────┘                    └────────────────────────┘
                   │                                             │
                   └─────────────────┐         ┌─────────────────┘
                                     ▼         ▼
                            ┌─────────────────────────┐
                            │ Behavioral Analyzer (3) │  <-- Random Forest (Shared TF-IDF Space)
                            └─────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │  Anomaly Detector (4)   │  <-- Local Outlier Factor (Shared LOF)
                            └─────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    Threat Scorer &      │  <-- Aggregates numerical outputs &
                            │    Decision Engine      │      decides (ALLOW/FLAG/SANITIZE/BLOCK)
                            └─────────────────────────┘
                                         │
                                 (If SAFE/SANITIZE)
                                         │
                                         ▼
                                   [ LLM Service ]
                                         │
                                         ▼
                                 [ Output Scrutiny ]
                                         │
                                         ▼
                                  [ User Response ]
```
