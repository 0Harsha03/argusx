"""ArgusX — Application Configuration (Pydantic Settings)"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    APP_NAME: str = "ArgusX"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION-USE-32-CHAR-SECRET"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./argusx.db"
    # For Postgres: postgresql+asyncpg://user:pass@localhost:5432/argusx

    # ── Redis (optional) ──────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False

    # ── Model Paths ───────────────────────────────────────────────────────────
    MODELS_DIR: str = "app/models/artifacts"
    BEHAVIORAL_MODEL_PATH: str = "app/models/artifacts/behavioral_model.pkl"
    ANOMALY_DETECTOR_PATH: str = "app/models/artifacts/anomaly_detector.pkl"
    VECTORIZER_PATH: str = "app/models/artifacts/vectorizer.pkl"
    DISTILBERT_MODEL_PATH: str = "models/distilbert_pi"

    # ── Threat Scoring Thresholds ─────────────────────────────────────────────
    SCORE_BLOCK_THRESHOLD: float = 75.0
    SCORE_FLAG_THRESHOLD: float = 50.0
    SCORE_SANITIZE_THRESHOLD: float = 35.0   # raised 30→35 (eval: eliminates 20 benign FPs)

    # Weights for final score composition (must sum to 1.0)
    WEIGHT_PATTERN: float = 0.30
    WEIGHT_SEMANTIC: float = 0.25
    WEIGHT_BEHAVIORAL: float = 0.30
    WEIGHT_ANOMALY: float = 0.15

    # ── Strategy D: Dynamic Threshold via Triple-Engine Agreement ─────────────
    # Lowers the SANITIZE threshold when Pattern, Semantic, and Behavioral
    # engines show corroborating evidence — targeting threshold-suppressed
    # prompt injection attacks that have no explicit regex vocabulary.
    #
    # Audit basis (2026-06-21):
    #   52% of injection FNs have final_score in [30, 35) — threshold suppression
    #   84.6% of MITRE TPs are driven by Behavioral+Semantic (not regex)
    #   Hard Refusal Rate on MITRE_FRR = 0.13% — invariant across strategies
    #
    # Set STRATEGY_D_ENABLED=false to revert to baseline (Strategy A).
    STRATEGY_D_ENABLED: bool = True

    # Signal thresholds — minimum score for each engine to be considered
    # an "elevated signal" eligible to participate in agreement.
    STRATEGY_D_PAT_SIGNAL_MIN: float = 50.0   # Pattern Engine fire weight ≥ 50
    STRATEGY_D_SEM_SIGNAL_MIN: float = 35.0   # SBERT cosine similarity × 100 ≥ 35
    STRATEGY_D_BEH_SIGNAL_MIN: float = 70.0   # Random Forest confidence ≥ 70

    # SANITIZE thresholds applied when agreement conditions are met:
    #   Triple agreement (Pat + Sem + Beh): strongest evidence — deepest reduction
    #   Dual agreement (Sem + Beh only):   most common injection FN profile
    #   Single Beh:                         conservative reduction, minimal risk
    STRATEGY_D_THR_TRIPLE: float = 28.0   # Pat ≥ 50 AND Sem ≥ 35 AND Beh ≥ 70
    STRATEGY_D_THR_SEM_BEH: float = 31.0  # Sem ≥ 35 AND Beh ≥ 70 (no pattern)
    STRATEGY_D_THR_BEH_ONLY: float = 33.0 # Beh ≥ 70 only (weakest reduction)

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
