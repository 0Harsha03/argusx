"""
ArgusX — Layer 4: Anomaly Detection Engine
==========================================
Uses the pre-trained LocalOutlierFactor model (anomaly_detector.pkl).

LOF was trained on the same TF-IDF feature space as the behavioral model,
so it detects structurally *novel* prompts that don't resemble anything
in the training corpus — a strong signal for zero-day / obfuscated attacks.

LOF quirk: sklearn's LOF .predict() returns -1 (outlier) or 1 (inlier).
Since it was fitted with novelty=False (fit-only mode from the prototype),
we use the decision_function score to derive a continuous anomaly score.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    score: float          # 0–100 (higher = more anomalous)
    is_anomaly: bool
    lof_score: float      # raw LOF decision function value (negative = more anomalous)
    explanation: str


class AnomalyDetector:
    """
    Wraps the pre-trained LOF model to detect structurally anomalous prompts.

    The LOF model in the prototype was trained with:
        LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    fitted via .fit() only (not novelty=True), so prediction is done via
    .negative_outlier_factor_ on training data. For inference on new data,
    we reconstruct a usable score from the fitted model's kneighbors.
    """

    # LOF score below this → treat as anomaly
    ANOMALY_THRESHOLD = -1.0

    def __init__(self, lof_model, vectorizer) -> None:
        self._lof = lof_model
        self._vectorizer = vectorizer
        self._fitted = self._verify_model()

    def _verify_model(self) -> bool:
        """Check the LOF model has been fitted."""
        try:
            _ = self._lof.negative_outlier_factor_
            return True
        except AttributeError:
            logger.warning("LOF model does not appear to be fitted.")
            return False

    def analyze(self, text: str) -> AnomalyDetectionResult:
        """
        Compute anomaly score for *text*.

        Strategy: Use kneighbors distance from the LOF model's training data.
        Greater distance → more anomalous → higher score.
        """
        if not self._fitted:
            return AnomalyDetectionResult(
                score=0.0, is_anomaly=False, lof_score=0.0,
                explanation="Anomaly detector not available.",
            )

        try:
            vec = self._vectorizer.transform([text]).toarray()

            # Get k-nearest neighbour distances to training data
            distances, _ = self._lof.kneighbors(vec, n_neighbors=min(5, self._lof.n_neighbors_))
            mean_dist = float(np.mean(distances))

            # Normalise distance to 0–100 score
            # Training data distances typically fall in [0, 2.0] for TF-IDF cosine
            # We clip at 1.5 to avoid outlier bias
            normalised = min(mean_dist / 1.5, 1.0)
            score = round(normalised * 100.0, 2)

            # Cross-check with negative outlier factors from training set
            nof_min  = float(np.min(self._lof.negative_outlier_factor_))
            nof_mean = float(np.mean(self._lof.negative_outlier_factor_))
            lof_score_proxy = -(mean_dist / max(abs(nof_mean), 1e-6))
            is_anomaly = score >= 50.0 or lof_score_proxy < self.ANOMALY_THRESHOLD

            explanation = (
                f"Prompt is {'anomalous' if is_anomaly else 'structurally normal'}. "
                f"Mean distance to training distribution: {mean_dist:.4f}. "
                f"Anomaly score: {score:.1f}/100."
            )

            return AnomalyDetectionResult(
                score=score,
                is_anomaly=is_anomaly,
                lof_score=round(lof_score_proxy, 4),
                explanation=explanation,
            )

        except Exception as exc:
            logger.error("Anomaly detection failed: %s", exc)
            return AnomalyDetectionResult(
                score=0.0, is_anomaly=False, lof_score=0.0,
                explanation=f"Anomaly detection error: {exc}",
            )
