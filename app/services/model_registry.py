"""
ArgusX — Model Registry
========================
Singleton that loads and holds all three pre-trained model artifacts:
  • behavioral_model.pkl   (RandomForestClassifier)
  • anomaly_detector.pkl   (LocalOutlierFactor — fitted)
  • vectorizer.pkl         (TfidfVectorizer — fitted)

Designed to be loaded once at startup via the FastAPI lifespan,
then attached to app.state.model_registry for zero-cost access.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Loads and exposes the three pre-trained model artifacts.
    Thread-safe after initial load — models are read-only at inference.
    """

    def __init__(self) -> None:
        self.behavioral_model = None
        self.anomaly_detector = None
        self.vectorizer       = None
        self._loaded          = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and all(
            m is not None
            for m in (self.behavioral_model, self.anomaly_detector, self.vectorizer)
        )

    def load_all(self) -> None:
        """
        Load all model artifacts from disk.
        Raises RuntimeError if any critical model fails to load.
        """
        self.vectorizer       = self._load(settings.VECTORIZER_PATH,       "vectorizer")
        self.behavioral_model = self._load(settings.BEHAVIORAL_MODEL_PATH, "behavioral_model")
        self.anomaly_detector = self._load(settings.ANOMALY_DETECTOR_PATH, "anomaly_detector")

        if self.vectorizer is None or self.behavioral_model is None:
            raise RuntimeError(
                "Critical model(s) failed to load. "
                "ArgusX cannot start without behavioral_model + vectorizer."
            )

        self._loaded = True
        logger.info(
            "ModelRegistry: all artifacts loaded | behavioral_model=%s | "
            "anomaly_detector=%s | vectorizer=%s",
            type(self.behavioral_model).__name__,
            type(self.anomaly_detector).__name__,
            type(self.vectorizer).__name__,
        )

    def _load(self, path: str, name: str):
        """Load a single pickle artifact, logging any failure."""
        abs_path = Path(path).resolve()
        if not abs_path.exists():
            logger.error("Model artifact not found: %s (resolved: %s)", path, abs_path)
            return None
        try:
            model = joblib.load(abs_path)
            logger.info("Loaded %s from %s (%d bytes)", name, abs_path, abs_path.stat().st_size)
            return model
        except Exception as exc:
            logger.error("Failed to load %s from %s: %s", name, abs_path, exc)
            return None

    def status(self) -> dict:
        return {
            "behavioral_model": self.behavioral_model is not None,
            "anomaly_detector": self.anomaly_detector is not None,
            "vectorizer":       self.vectorizer is not None,
        }
