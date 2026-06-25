"""
ArgusX v11 — Phase 2A: DistilBERT Semantic Analysis Engine
==========================================================
Replaces SBERT with a fine-tuned DistilBERT sequence classifier
specifically optimized for prompt injection detection.

Interface: Drop-in replacement for SBERTSemanticAnalyzer.
  - Input:  raw text string
  - Output: SemanticAnalysisResult (identical schema)
"""

import logging
import torch
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class SemanticAnalysisResult:
    """Identical schema to the TF-IDF and SBERT SemanticAnalysisResult."""
    score: float                        # 0–100
    max_similarity: float               # raw probability (0–1)
    top_match: Optional[str] = None     
    above_threshold: bool = False

class DistilBERTSemanticAnalyzer:
    """
    DistilBERT-powered semantic analyzer (drop-in replacement for SBERTSemanticAnalyzer).

    Tokenizes the incoming prompt and passes it through DistilBERT to compute
    the probability of Prompt Injection (class 1).

    The interface is identical to SBERTSemanticAnalyzer:
        .analyze(text: str) -> SemanticAnalysisResult
    """

    SIMILARITY_THRESHOLD = 0.35   # 35% probability triggers above_threshold

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the DistilBERT model from global ModelRegistry."""
        from app.services.model_registry import global_registry
        
        if global_registry and global_registry.distilbert_model is not None and global_registry.distilbert_tokenizer is not None:
            logger.info("DistilBERTSemanticAnalyzer reusing pre-loaded DistilBERT from ModelRegistry")
            self._model = global_registry.distilbert_model
            self._tokenizer = global_registry.distilbert_tokenizer
        else:
            raise RuntimeError("DistilBERT model is not loaded in ModelRegistry. Cannot initialize.")

    def analyze(self, text: str) -> SemanticAnalysisResult:
        """
        Tokenize *text* and pass through DistilBERT.

        Returns:
            SemanticAnalysisResult with score in [0, 100].
        """
        if self._model is None or self._tokenizer is None:
            logger.warning("DistilBERT model unavailable — returning 0 semantic score.")
            return SemanticAnalysisResult(score=0.0, max_similarity=0.0)

        try:
            # Determine runtime device dynamically
            device = next(self._model.parameters()).device
            
            inputs = self._tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move tokenizer tensors to the exact same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Execute inside no_grad. Model is already guaranteed to be in eval() mode.
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                p = probs[0, 1].item()  # Probability of class 1 (Prompt Injection)
                
            score = min(p * 100.0, 100.0)
            
            logger.debug("DistilBERT inference completed: score=%.2f", score)
            
            return SemanticAnalysisResult(
                score=round(score, 2),
                max_similarity=round(p, 4),
                top_match="DistilBERT PI Prediction",
                above_threshold=(p >= self.SIMILARITY_THRESHOLD)
            )
            
        except Exception as exc:
            logger.error("DistilBERT semantic analysis failed: %s", exc)
            return SemanticAnalysisResult(score=0.0, max_similarity=0.0)
