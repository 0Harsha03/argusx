# ArgusX v11 — Production Runtime Compatibility Audit

## Objective
To determine if the production runtime can safely transition from the legacy `DetectionPipeline` to the `AdaptiveDetectionPipeline` without breaking external behavior, downstream clients, or database persistence logic.

---

## 1. Interface Compatibility

*   **Constructors:** Identical. Both accept precisely `(registry: ModelRegistry)`.
*   **Analyze Method:** Identical signature. Both accept `(raw_prompt: str, session_id: Optional[str] = None, source_ip: Optional[str] = None) -> dict`.
*   **Return Objects:** Highly compatible. `DetectionPipeline` returns a dictionary of 18 specific keys. `AdaptiveDetectionPipeline` returns the exact same 18 keys with identical types, plus two new diagnostic extensions (`selected_route`, `route_metadata`). 
*   **Exceptions & Dependencies:** Both pipelines bubble up standard Python `Exception` types when underlying sub-modules fail, which are handled uniformly by the FastAPI route layer. Neither introduces new external dependencies.

---

## 2. Global Codebase Impact
**Would changing `dependencies.py` require changes anywhere else?**
**No.** 

Because the `AdaptiveDetectionPipeline` was explicitly engineered as a "drop-in replacement," modifying `app/api/dependencies.py` to instantiate the new pipeline requires zero changes to route handlers or evaluation scripts. 

---

## 3. API Response Consistency
**Would `POST /protect` produce the exact same JSON schema?**
**Yes.**

*   **Response Model:** The endpoint uses the `ProtectResponse` Pydantic model. The `analysis` field is typed as `Dict[str, Any]`. 
*   **Serialization:** Pydantic will safely serialize the two new diagnostic keys (`selected_route`, `route_metadata`) dynamically. 
*   **HTTP Status Codes & Error Payloads:** Remain exactly the same, governed by FastAPI exception handlers wrapping the pipeline.

---

## 4. Logging and Persistence
*   **Logging:** The console log messages output by the pipeline itself change slightly (e.g., adding `"v9 analysis complete"` and printing the `route` selected). Because `LOG_JSON=True` is enabled, structured logging downstream processors will seamlessly ingest the new data without parsing breakage.
*   **AnalysisLog Persistence:** In `protect.py`, the `AnalysisLog` SQLAlchemy model explicitly unpacks named keys from the `analysis` dictionary (e.g., `pattern_score=analysis["pattern_score"]`). Because the dictionary keys are identical, database insertion succeeds. The two new keys are simply ignored by the ORM mapping, preventing any database schema errors.

---

## 5. Middleware Compatibility
Middleware operates completely independent of the detection pipeline. Rate limiting, GZIP, CORS, Authentication, and Lifespan events are fully isolated. The pipeline is purely an algorithmic processing block. Middleware behavior is **100% unchanged**.

---

## 6. Exception Handling
Both pipelines rely on the exact same error-handling design. Any internal crashes during `pipeline.analyze()` are caught by the `try...except Exception` block in `protect.py` and safely converted to a standardized `HTTP 500` error. The FastAPI error boundaries remain entirely intact.

---

## 7. Downstream Integrations
*   **Frontend/REST Clients:** Standard JSON parsers ignore unknown fields, so the two new keys will not break frontend states.
*   **Swagger/OpenAPI:** The OpenAPI schema for the `analysis` field is simply `object` (`Dict[str, Any]`), meaning the contract remains unbroken.
*   **Evaluation Scripts:** Evaluation loops that assert on `"final_score"` or `"decision"` will continue to operate flawlessly because those keys are guaranteed.

---

## 8. Functionality Assessment
The `AdaptiveDetectionPipeline` perfectly encapsulates the entire responsibility of the legacy `DetectionPipeline`. It orchestrates the preprocessor, Pattern Engine, Semantic Engine (via Router), Behavioral RF, LOF Anomaly Detector, and the Threat Scorer. There is zero missing functionality.

---

## 9. Risk Assessment

*   **API Compatibility Risk:** **LOW**. Pydantic safely handles `Dict[str, Any]` dynamic supersets.
*   **Runtime Compatibility Risk:** **LOW**. The execution timeline and synchronous execution environment remain identical.
*   **Deployment Risk:** **LOW**. Both pipelines already successfully build against the existing Docker/environment constraints.
*   **Regression Risk:** **LOW**. Architecturally, the routing layer safely proxies the exact same inputs and outputs. 

*(Note: While the architectural swap is low risk, the algorithmic consequences of introducing DistilBERT inference carries isolated risk, as noted in previous audits. But structurally, the pipeline wrapper is 100% safe).*

---

## 10. Final Recommendation

**Option A: AdaptiveDetectionPipeline is a drop-in replacement.**

The pipelines are structurally, programmatically, and behaviorally compatible at the interface level. Only `dependencies.py` requires modification to execute the pipeline transition. The repository is cleared for hot-swapping.
