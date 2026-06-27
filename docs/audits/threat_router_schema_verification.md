# ArgusX v11 — ThreatRouter Schema Verification Audit

## Objective
To strictly verify whether the schema mismatches identified between the `PatternDetector` and `ThreatRouter` are genuine runtime bugs, or if they are resolved by intentional intermediate normalizations hidden elsewhere in the repository.

---

## 1. Emitted Category Strings
At runtime, `PatternDetector` iterates through its `_RULES` list. When a regex match occurs, it extracts the exact string defined in `PatternRule.category` and appends it to `pattern_result.details`.

By inspecting `app/detection/pattern_detector.py`, the exhaustive list of emitted categories is:
*   `INSTRUCTION_OVERRIDE`
*   `JAILBREAK`
*   `ROLE_MANIPULATION`
*   `DATA_EXFILTRATION`
*   `PROMPT_INJECTION`
*   `ADVERSARIAL_INPUT`
*   `MALWARE_GENERATION`
*   `CREDENTIAL_THEFT`
*   `EXPLOITATION`
*   `CYBER_ABUSE`

---

## 2. Intermediate Transformations
**Is there any intermediate mapping or alias resolution?**
**No.** 

The `AdaptiveDetectionPipeline` (Layer orchestrator) extracts the categories using a direct set comprehension:
```python
pattern_categories = list({d["category"] for d in pattern_result.details})
```
It then passes this exact list directly to `ThreatRouter.route()`. 

The *only* transformation that occurs in the entire pipeline happens inside `ThreatRouter._select_route()`, where it applies `.lower()`:
```python
detected = frozenset(c.lower() for c in pattern_categories)
```

---

## 3. Execution Trace
1.  **PatternDetector:** Emits `["MALWARE_GENERATION"]`.
2.  **AdaptiveDetectionPipeline:** Passes `["MALWARE_GENERATION"]` directly to the router.
3.  **ThreatRouter:** Converts to `{"malware_generation"}`.
4.  **ThreatRouter:** Computes `{"malware_generation"} & _CYBER_CATEGORIES`.

---

## 4. Mismatch Verification
Every mismatch identified in the previous audit **reaches runtime unchanged (Option A)**. There is no alias resolution.
*   **Malware:** Evaluated as `"malware_generation"`. The router expects `"malware"`. Intersection = `False`.
*   **Phishing / DDoS:** Evaluated as `"cyber_abuse"`. The router expects `"phishing"` and `"ddos"`. Intersection = `False`.
*   **Code Execution:** Evaluated as `"adversarial_input"`. The router expects `"remote_code_execution"`. Intersection = `False`.

---

## 5. Simulated Routing Path

| Attack Type | Emitted Category | After Transformation | Router Decision | Final Route |
| :--- | :--- | :--- | :--- | :--- |
| Malware generation | `MALWARE_GENERATION` | `malware_generation` | Miss | **Prompt Injection** *(Bug)* |
| Phishing email | `CYBER_ABUSE` | `cyber_abuse` | Miss | **Prompt Injection** *(Bug)* |
| DDoS request | `CYBER_ABUSE` | `cyber_abuse` | Miss | **Prompt Injection** *(Bug)* |
| System prompt extraction | `DATA_EXFILTRATION` | `data_exfiltration` | Match | **Cyber Threat** *(Bug)* |
| Jailbreak | `JAILBREAK` | `jailbreak` | Miss | **Prompt Injection** *(Correct)* |
| Credential theft | `CREDENTIAL_THEFT` | `credential_theft` | Match | **Cyber Threat** *(Correct)* |

---

## 6. Audit Conclusion Verdict
The previous audit's conclusions are **FULLY CORRECT**.

There is no hidden normalization logic protecting the router. The `_CYBER_CATEGORIES` set in `threat_router.py` has severely drifted from the actual labels emitted by the `PatternDetector`. This schema drift is a critical runtime bug that will cause severe routing failures (and consequently, severe detection failures) if the Adaptive pipeline is activated in its current state.

---

## 7. Final Recommendation

**Option A: The routing schema mismatch is a genuine runtime bug.**

`ThreatRouter` and `PatternDetector` must be modified to align their schemas before the `AdaptiveDetectionPipeline` is activated in production (Phase 2B).
