"""Cross-platform configuration for the defensive AI service."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_NAME = "TCHS Defensive AI Service"
SERVICE_VERSION = "0.1.0"
DATASET_NAME = "defense-v1"

OBJECT_CLASSES: tuple[str, ...] = (
    "defensive_end",
    "defensive_tackle",
    "middle_linebacker",
    "inside_linebacker",
    "outside_linebacker",
    "cornerback",
    "safety",
    "football",
    "official",
)

DERIVED_FOOTBALL_LABELS: tuple[str, ...] = (
    "defensive_front",
    "box_count",
    "coverage_shell",
    "blitz_look",
    "corner_leverage",
    "safety_rotation",
)

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings populated from AI_SERVICE_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=AI_SERVICE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_host: str = "127.0.0.1"
    service_port: int = 8000
    dataset_dir: Path = AI_SERVICE_DIR / "datasets" / DATASET_NAME
    model_dir: Path = AI_SERVICE_DIR / "models"
    detector_model: str = "yolo11n.pt"
    detector_confidence: float = Field(default=0.35, ge=0.01, le=1.0)
    detector_iou: float = Field(default=0.50, ge=0.01, le=1.0)
    detector_device: str = "auto"
    detector_max_detections: int = Field(default=100, ge=1, le=500)
    allow_model_download: bool = False

    @field_validator("dataset_dir", "model_dir", mode="after")
    @classmethod
    def resolve_service_relative_path(cls, value: Path) -> Path:
        """Resolve relative overrides from the ai-service directory."""
        return value if value.is_absolute() else AI_SERVICE_DIR / value


settings = Settings()
