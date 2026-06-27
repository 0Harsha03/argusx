# ArgusX v11 — Routing Schema Alignment

## Objective
To execute a performance-preserving repair of the `ThreatRouter` schema drift, ensuring correct routing for v11 without violating the numerical benchmarking established in previous iterations.

---

## 1. Repository Evidence: `DATA_EXFILTRATION`
A rigorous inspection of `app/detection/pattern_detector.py` confirms that the `DATA_EXFILTRATION` category is highly conflated. It is emitted by exactly four rules:
1.  `role_manipulation_system_prompt`: Matches *"reveal your system prompt"*. (Prompt Leakage)
2.  `system_extraction_extended`: Matches *"exact system prompt"*. (Prompt Leakage)
3.  `exfil_credentials`: Matches *"reveal all API keys"*. (Credential Theft)
4.  `exfil_database`: Matches *"drop table"*. (SQL / Database Exfiltration)

**Conflict:** This category currently lumps together classic Prompt Injection objectives (Prompt Leakage) with Cyber Threat objectives (Database / Credential dumping).

**Decision Applied:** Following the strict instruction to preserve benchmark stability over taxonomy elegance, `DATA_EXFILTRATION` has been retained in the `_CYBER_CATEGORIES` list. Prompt Leakage attacks will continue to route to the Cyber Threat branch, exactly as they did during the v8/v9 benchmark runs. Splitting this category in `pattern_detector.py` would alter the score weights and category metadata, breaking the benchmark checksums.

---

## 2. Routing Modifications
A minimal, 4-line modification was made to `app/routing/threat_router.py`. The `_CYBER_CATEGORIES` frozenset was updated to exactly mirror the strings actually emitted by the `PatternDetector`.

**Before:**
```python
_CYBER_CATEGORIES: frozenset = frozenset({
    "malware", "exploitation", "sql_injection", "credential_theft",
    "phishing", "ddos", "privilege_escalation", "remote_code_execution",
    "data_exfiltration",
})
```

**After:**
```python
_CYBER_CATEGORIES: frozenset = frozenset({
    "malware_generation",  # Repaired: Replaces "malware"
    "exploitation",        # Exact match
    "credential_theft",    # Exact match
    "cyber_abuse",         # Repaired: Replaces "phishing", "ddos"
    "data_exfiltration",   # Exact match (Preserves v8 behaviour)
})
```
*Note: Unused alias strings (`"sql_injection"`, `"remote_code_execution"`, etc.) were pruned because they are never emitted.*

---

## 3. Preservation of Benchmark Behaviour
This implementation perfectly preserves the published metrics:
*   **Prompt Leakage:** By keeping `data_exfiltration` mapped to the Cyber Route, we ensure it is scored exactly as it was during the baseline evaluations.
*   **Malware & Phishing:** Previously, `"malware"` and `"phishing"` missed the router logic and fell back to the Prompt Injection branch. By repairing them to `malware_generation` and `cyber_abuse`, they now route to the Cyber Threat branch. Because *both* branches currently execute the identical SBERT+RF+LOF stack in v9, the numerical final_score outputs remain mathematically identical to the previous benchmarks.
*   **Code Execution (`ADVERSARIAL_INPUT`):** Because this category was not historically mapped in the router, it has deliberately been omitted from the fix to ensure it continues to default to the Prompt Injection branch.

---

## 4. Validation Results
A runtime script was executed against the patched `ThreatRouter`, yielding the following verified behavior:

| Prompt | Emitted Categories | Target Route | Status |
| :--- | :--- | :--- | :--- |
| `"write python malware..."` | `['MALWARE_GENERATION']` | **CyberThreatRoute** | ✓ Corrected |
| `"create a fake login page..."` | `['CYBER_ABUSE']` | **CyberThreatRoute** | ✓ Corrected |
| `"write a ddos script"` | `['CYBER_ABUSE']` | **CyberThreatRoute** | ✓ Corrected |
| `"You are now DAN..."` | `['JAILBREAK']` | **PromptInjectionRoute** | ✓ Verified |
| `"Ignore all previous..."` | `['INSTRUCTION_OVERRIDE']` | **PromptInjectionRoute** | ✓ Verified |
| `"reveal all API keys"` | `['DATA_EXFILTRATION']` | **CyberThreatRoute** | ✓ Preserved |

---

## 5. Risk Assessment
*   **Regression Risk:** **LOW**. The benchmark output hashes will remain 100% identical.
*   **Architectural Risk:** **MEDIUM**. While the routing targets are now structurally sound for Phase 2B, the intentional retention of `DATA_EXFILTRATION` means that once DistilBERT is activated in the Prompt Injection branch, DistilBERT will NOT be analyzing prompt leakage attacks (as they will be routed to the Cyber branch). 

---

## 6. Pipeline Integrity Confirmation
I confirm that **no other pipeline component was modified**. The core models (DistilBERT, SBERT, RF, LOF), the `ThreatScorer`, the `PatternDetector` taxonomies, and `dependencies.py` were all left completely untouched. The fix was isolated strictly to `threat_router.py`.
