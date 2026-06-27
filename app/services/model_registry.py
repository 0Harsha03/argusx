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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

global_registry = None

class ModelRegistry:
    """
    Loads and exposes the three pre-trained model artifacts.
    Thread-safe after initial load — models are read-only at inference.
    """

    def __init__(self) -> None:
        global global_registry
        global_registry = self
        
        self.behavioral_model = None
        self.anomaly_detector = None
        self.vectorizer       = None
        self.sbert_model      = None
        self.sbert_anchor_embeddings = None
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self.platt_db         = None
        self.platt_rf         = None
        self.router_vectorizer = None
        self.router_classifier = None
        self._loaded          = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and all(
            m is not None
            for m in (
                self.behavioral_model, 
                self.anomaly_detector, 
                self.vectorizer,
                self.sbert_model,
                self.distilbert_model,
                self.platt_db,
                self.platt_rf,
                self.router_vectorizer,
                self.router_classifier
            )
        )

    def load_all(self) -> None:
        """
        Load all model artifacts from disk and HuggingFace.
        Raises RuntimeError if any critical model fails to load.
        """
        self.vectorizer       = self._load_pickle(settings.VECTORIZER_PATH, "vectorizer")
        self.behavioral_model = self._load_pickle(settings.BEHAVIORAL_MODEL_PATH, "behavioral_model")
        self.anomaly_detector = self._load_pickle(settings.ANOMALY_DETECTOR_PATH, "anomaly_detector")
        self.platt_db         = self._load_pickle(settings.PLATT_DB_PATH, "platt_db")
        self.platt_rf         = self._load_pickle(settings.PLATT_RF_PATH, "platt_rf")
        
        # Load semantic router models
        self.router_vectorizer = self._load_pickle("models/router_vec_D.pkl", "router_vectorizer")
        self.router_classifier = self._load_pickle("models/router_clf_D.pkl", "router_classifier")
        
        self._load_sbert()
        self._load_distilbert()

        if self.vectorizer is None or self.behavioral_model is None:
            raise RuntimeError(
                "Critical model(s) failed to load. "
                "ArgusX cannot start without behavioral_model + vectorizer."
            )
        if self.platt_db is None or self.platt_rf is None:
            raise RuntimeError(
                "Calibration models failed to load. "
                "ArgusX cannot start without Platt scaling artifacts."
            )
        if self.router_vectorizer is None or self.router_classifier is None:
            raise RuntimeError(
                "Semantic router models failed to load. "
                "ArgusX cannot start without the semantic router."
            )
        if self.sbert_model is None or self.distilbert_model is None:
            raise RuntimeError(
                "Critical semantic model(s) failed to load. "
                "ArgusX cannot start without SBERT and DistilBERT."
            )

        self._loaded = True
        logger.info(
            "ModelRegistry: all artifacts loaded | behavioral_model=%s | "
            "anomaly_detector=%s | vectorizer=%s | sbert=%s | distilbert=%s | platt_db=%s | platt_rf=%s",
            type(self.behavioral_model).__name__,
            type(self.anomaly_detector).__name__,
            type(self.vectorizer).__name__,
            type(self.sbert_model).__name__,
            type(self.distilbert_model).__name__,
            type(self.platt_db).__name__,
            type(self.platt_rf).__name__,
            type(self.router_classifier).__name__,
        )

    def _load_pickle(self, path: str, name: str):
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

    def _load_sbert(self) -> None:
        try:
            logger.info("Loading SBERT model: all-MiniLM-L6-v2")
            self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Pre-compute anchor embeddings exactly as SBERTSemanticAnalyzer does
            from app.detection.sbert_semantic_analyzer import SBERT_ADVERSARIAL_ANCHORS
            self.sbert_anchor_embeddings = self.sbert_model.encode(
                SBERT_ADVERSARIAL_ANCHORS,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            logger.info("SBERT loaded successfully with anchor embeddings")
        except Exception as exc:
            logger.error("Failed to load SBERT: %s", exc)
            self.sbert_model = None
            self.sbert_anchor_embeddings = None

    def _load_distilbert(self) -> None:
        try:
            model_path = Path(settings.DISTILBERT_MODEL_PATH).resolve()
            logger.info("Loading DistilBERT model from: %s", model_path)
            if not model_path.exists():
                logger.error("DistilBERT model path does not exist: %s", model_path)
                return
                
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            
            # Optimize for inference
            self.distilbert_model.eval()
            if torch.cuda.is_available():
                self.distilbert_model = self.distilbert_model.to("cuda")
                
            logger.info("DistilBERT loaded successfully")
        except Exception as exc:
            logger.error("Failed to load DistilBERT: %s", exc)
            self.distilbert_model = None
            self.distilbert_tokenizer = None

    def status(self) -> dict:
        return {
            "behavioral_model": self.behavioral_model is not None,
            "anomaly_detector": self.anomaly_detector is not None,
            "vectorizer":       self.vectorizer is not None,
            "sbert_model":      self.sbert_model is not None,
            "distilbert_model": self.distilbert_model is not None,
            "platt_db":         self.platt_db is not None,
            "platt_rf":         self.platt_rf is not None,
            "router_vectorizer": self.router_vectorizer is not None,
            "router_classifier": self.router_classifier is not None,
        }
