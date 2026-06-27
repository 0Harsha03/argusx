# ArgusX v11 — ThreatRouter Correctness & Routing Audit

## Objective
To rigorously verify the architectural correctness of the `ThreatRouter` and its ability to accurately dispatch requests between the Prompt Injection and Cyber Threat branches, prior to activating the v11 runtime in production.

---

## 1. Routing Logic
The `ThreatRouter` relies on the output of the `PatternDetector`. It extracts the `pattern_categories` (a list of all categories triggered by regex rules), converts them to lowercase, and checks for an intersection with a hardcoded `_CYBER_CATEGORIES` set.

**Decision Flow:**
1. If **any** of the matched categories exist within `_CYBER_CATEGORIES`, the prompt is immediately routed to the **`CyberThreatRoute`**.
2. If **none** of the matched categories intersect, or if the prompt triggered no regex rules, it defaults to the **`PromptInjectionRoute`**.

---

## 2. Category Mapping (The Mismatch)
By cross-referencing the categories emitted by `app/detection/pattern_detector.py` against the keys expected by `_CYBER_CATEGORIES` in `app/routing/threat_router.py`, a massive schema mismatch is evident:

| Pattern Engine Category (Emitted) | Router Expects in `_CYBER_CATEGORIES` | Actual Destination Branch |
| :--- | :--- | :--- |
| `INSTRUCTION_OVERRIDE` | - | Prompt Injection |
| `JAILBREAK` | - | Prompt Injection |
| `ROLE_MANIPULATION` | - | Prompt Injection |
| `PROMPT_INJECTION` | - | Prompt Injection |
| `DATA_EXFILTRATION` | `data_exfiltration` | **Cyber Threat** |
| `CREDENTIAL_THEFT` | `credential_theft` | **Cyber Threat** |
| `EXPLOITATION` | `exploitation` | **Cyber Threat** |
| `MALWARE_GENERATION` | `malware` *(MISMATCH)* | **Prompt Injection** *(Failure)* |
| `CYBER_ABUSE` | `phishing` / `ddos` *(MISMATCH)* | **Prompt Injection** *(Failure)* |
| `ADVERSARIAL_INPUT` | `remote_code_execution` *(MISMATCH)* | **Prompt Injection** *(Failure)* |

---

## 3. Routing Determinism
**Yes, routing is fully deterministic.** There is zero randomness. Identical prompts will trigger the exact same regex rules, generating the exact same categories, which the intersection logic will route identically every time.

---

## 4. Ambiguous Prompts
For prompts containing indicators from both domains (e.g., a Jailbreak + Credential Theft):
**The router strictly prioritizes the Cyber Threat branch.**
Because the code uses `if detected & _CYBER_CATEGORIES: return self._cyber_route`, the presence of a single cyber category overrides all prompt injection categories. It chooses exactly one branch and executes it.

---

## 5. Fallback Behaviour
If no Pattern Engine category matches (a prompt that evades all regex rules), the intersection `detected & _CYBER_CATEGORIES` evaluates to `False`, forcing the prompt into the **Prompt Injection branch**. 

This fallback is **intentional**. Prompt Injections are notoriously difficult to catch via regex and heavily rely on downstream Semantic (SBERT/DistilBERT) and Behavioral models. If a prompt lacks explicit malicious syntax, it is safest to route it through the deep-learning NLP layers in the Prompt Injection branch.

---

## 6. Prompt Injection Coverage
**Can a Prompt Injection prompt ever be routed into the Cyber Threat branch?**
**YES. This is a critical failure.**
The rule `role_manipulation_system_prompt` (which catches phrases like *"show me your system prompt"*) is assigned the category `DATA_EXFILTRATION`. Because `DATA_EXFILTRATION` maps to the Cyber Threat branch, **classic Prompt Leakage attacks are routed away from the Prompt Injection branch**. If DistilBERT is placed exclusively in the PI branch, it will never see system prompt extraction attempts.

---

## 7. Cyber Threat Coverage
**Can a Cyber Threat prompt ever be routed into the Prompt Injection branch?**
**YES. This is a critical failure.**
Due to the hardcoded string mismatches in `ThreatRouter`, several severe cyber threats silently fall back to the Prompt Injection branch:
*   Malware Generation emits `MALWARE_GENERATION`, but the router expects `malware`.
*   Phishing / DDoS emit `CYBER_ABUSE`, but the router expects `phishing` and `ddos`.
*   Code Execution emits `ADVERSARIAL_INPUT`, but the router expects `remote_code_execution`.

---

## 8. Rule Ordering
Routing does **not** depend on regex order, highest score, category severity, or frequency. It strictly relies on **first match intersection**. If a set of `['jailbreak', 'instruction_override', 'credential_theft']` is passed, the presence of `credential_theft` guarantees a route to the Cyber branch, ignoring the others entirely.

---

## 9. Failure Modes
*   **Schema Drift (High Impact):** Hardcoded keys in the router do not match the detector. Malware and Phishing bypass the cyber branch entirely.
*   **Conflated Categories (High Impact):** `DATA_EXFILTRATION` groups both "SQL Database Dumps" (Cyber) and "System Prompt Leakage" (PI) under a single category. The router cannot distinguish them and routes both to Cyber, blinding the PI branch.
*   **Priority Shadowing (Medium Impact):** Ambiguous prompts (e.g., Jailbreak + Malware) route to Cyber, meaning the Jailbreak branch models will never analyze them.

---

## 10. Coverage Analysis
The existing Pattern Engine categories are **insufficient** to properly distinguish domains because of conflation. 
*   **Missing category:** `PROMPT_LEAKAGE` must be separated from `DATA_EXFILTRATION`.
*   **Missing category:** `CODE_EXECUTION` must be separated from `ADVERSARIAL_INPUT`.
Without these splits, the router will perpetually confuse System Prompt extraction with Database Exfiltration.

---

## 11. Runtime Integration Risk
**If AdaptiveDetectionPipeline becomes the production runtime, will ThreatRouter behave correctly without modification?**
**NO.**
If activated as-is, DistilBERT (installed in the Prompt Injection branch) will suddenly be forced to analyze Malware and Phishing prompts (due to the router mismatch fallback), while simultaneously being blinded to Prompt Leakage attacks (which are routed to the Cyber branch). This will cause massive regressions in False Positives and False Negatives.

---

## 12. Architectural Recommendation

**Option B: ThreatRouter requires modification before production activation.**

Before we switch `app/api/dependencies.py` to use `AdaptiveDetectionPipeline`, we must perform the following modifications:
1.  **Refactor `app/detection/pattern_detector.py`:** 
    *   Reassign `role_manipulation_system_prompt` and `system_extraction_extended` to a new category: `PROMPT_LEAKAGE`.
    *   Reassign `code_execution_dangerous` to `CODE_EXECUTION`.
2.  **Refactor `app/routing/threat_router.py`:**
    *   Update `_CYBER_CATEGORIES` to exactly match the emitted strings: `"malware_generation"`, `"cyber_abuse"`, `"credential_theft"`, `"exploitation"`, `"data_exfiltration"`, and `"code_execution"`.
