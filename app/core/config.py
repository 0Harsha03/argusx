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

    # ── Threat Scoring Thresholds ─────────────────────────────────────────────
    SCORE_BLOCK_THRESHOLD: float = 75.0
    SCORE_FLAG_THRESHOLD: float = 50.0
    SCORE_SANITIZE_THRESHOLD: float = 35.0   # raised 30→35 (eval: eliminates 20 benign FPs)

    # Weights for final score composition (must sum to 1.0)
    WEIGHT_PATTERN: float = 0.30
    WEIGHT_SEMANTIC: float = 0.25
    WEIGHT_BEHAVIORAL: float = 0.30
    WEIGHT_ANOMALY: float = 0.15

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
