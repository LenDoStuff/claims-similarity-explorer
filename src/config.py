from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_EMBEDDING_MODELS_DIR = DEFAULT_MODELS_DIR / "embeddings"
DEFAULT_RERANKER_MODELS_DIR = DEFAULT_MODELS_DIR / "rerankers"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_APP_CONFIG_PATH = PROJECT_ROOT / "app_config.toml"
DEFAULT_COLLECTION_NAME = "claims"
DEFAULT_MODEL_KEY = "multilingual-e5-small"
DEFAULT_EMBEDDING_VERSION = "e5-v1"

COLUMN_MAPPING_FIELDS = [
    ("Claim ID", "claim_id", "claim_id", True),
    ("Claim description", "description", "description", True),
    ("Line of business", "line_of_business", "line_of_business", False),
    ("Claim type", "claim_type", "claim_type", False),
    ("Cause of loss", "cause_of_loss", "cause_of_loss", False),
    ("Damaged object", "damaged_object", "damaged_object", False),
    ("Country", "country", "country", False),
    ("Claim status", "claim_status", "claim_status", False),
    ("Loss date", "loss_date", "loss_date", False),
    ("Reserve amount", "reserve_amount", "reserve_amount", False),
    ("Paid amount", "paid_amount", "paid_amount", False),
    ("Currency", "currency", "currency", False),
    ("Policy type", "policy_type", "policy_type", False),
]


@dataclass(frozen=True)
class EmbeddingModelConfig:
    key: str
    label: str
    repo_id: str
    model_dir: Path
    notes: str


@dataclass(frozen=True)
class RerankerModelConfig:
    key: str
    label: str
    repo_id: str
    model_dir: Path


@dataclass(frozen=True)
class ColumnConfig:
    claim_id: str = "claim_id"
    description: str = "claim_description"
    line_of_business: str = "line_of_business"
    claim_type: str = "claim_type"
    cause_of_loss: str = "cause_of_loss"
    damaged_object: str = "damaged_object"
    country: str = "country"
    claim_status: str = "claim_status"
    loss_date: str = "loss_date"
    reserve_amount: str = "reserve_amount"
    paid_amount: str = "paid_amount"
    currency: str = "currency"
    policy_type: str = "policy_type"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ColumnConfig":
        defaults = cls()
        return cls(
            **{
                attr: str(values.get(config_key, getattr(defaults, attr))).strip()
                for _, config_key, attr, _ in COLUMN_MAPPING_FIELDS
            }
        )

    def validate_required(self) -> None:
        missing = [
            config_key
            for _, config_key, attr, required in COLUMN_MAPPING_FIELDS
            if required
            if not getattr(self, attr)
        ]
        if missing:
            raise ValueError(f"Required column mapping(s) cannot be blank: {', '.join(f'columns.{key}' for key in missing)}")

    @property
    def selected_columns(self) -> list[str]:
        seen: set[str] = set()
        ordered = [
            self.claim_id,
            self.description,
            self.line_of_business,
            self.claim_type,
            self.cause_of_loss,
            self.damaged_object,
            self.country,
            self.claim_status,
            self.loss_date,
            self.reserve_amount,
            self.paid_amount,
            self.currency,
            self.policy_type,
        ]
        return [col for col in ordered if col and not (col in seen or seen.add(col))]

    @property
    def embedding_columns(self) -> list[str]:
        columns = [
            self.description,
            self.line_of_business,
            self.claim_type,
            self.cause_of_loss,
            self.damaged_object,
            self.country,
        ]
        return [column for column in columns if column]

    def mapping_rows(self) -> list[dict[str, str]]:
        rows = []
        for label, config_key, attr, required in COLUMN_MAPPING_FIELDS:
            source_column = getattr(self, attr)
            rows.append(
                {
                    "role": label,
                    "config_key": f"columns.{config_key}",
                    "source_column": source_column or "(skipped)",
                    "required": "yes" if required else "no",
                }
            )
        return rows


@dataclass(frozen=True)
class AppConfig:
    snowflake_table: str = ""
    snowflake_row_limit: int | None = None
    model_key: str = DEFAULT_MODEL_KEY
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_version: str = DEFAULT_EMBEDDING_VERSION

    @classmethod
    def from_app_config(cls) -> "AppConfig":
        if not DEFAULT_APP_CONFIG_PATH.exists():
            raise RuntimeError(f"App config file was not found: {DEFAULT_APP_CONFIG_PATH}")
        with DEFAULT_APP_CONFIG_PATH.open("rb") as file:
            app_data = tomllib.load(file)
        snowflake = app_data.get("snowflake", {})
        columns = app_data.get("columns", {})
        row_limit = None
        if isinstance(snowflake, dict) and snowflake.get("row_limit") not in (None, ""):
            row_limit = int(snowflake["row_limit"])
        return cls(
            snowflake_table=str(snowflake.get("table", "")).strip() if isinstance(snowflake, dict) else "",
            snowflake_row_limit=row_limit,
            columns=ColumnConfig.from_mapping(columns if isinstance(columns, dict) else {}),
        )

    def validate_source(self) -> None:
        if not self.snowflake_table:
            raise ValueError("Required app config value cannot be blank: snowflake.table")
        if self.snowflake_row_limit is not None and self.snowflake_row_limit <= 0:
            raise ValueError("snowflake.row_limit must be a positive integer when set")
        self.columns.validate_required()

    def index_manifest_path_for_model(self, model_key: str) -> Path:
        return self.artifacts_dir / f"index_manifest_{model_storage_key(model_key)}.json"

    def clusters_path_for_model(self, model_key: str) -> Path:
        return self.artifacts_dir / f"clusters_{model_storage_key(model_key)}.json"

    def cluster_map_path_for_model(self, model_key: str) -> Path:
        return self.artifacts_dir / f"cluster_map_{model_storage_key(model_key)}.json"

    @property
    def cluster_reviews_path(self) -> Path:
        return self.artifacts_dir / "cluster_reviews.json"


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def model_storage_key(model_key: str) -> str:
    storage_key = re.sub(r"[^0-9A-Za-z_-]+", "_", model_key).strip("_")
    return storage_key.replace("-", "_").lower()


def versioned_collection_name(base_collection_name: str, model_key: str, index_hash: str) -> str:
    return f"{base_collection_name}_{model_storage_key(model_key)}_{index_hash[:12]}"


def embedding_models_dir() -> Path:
    return DEFAULT_EMBEDDING_MODELS_DIR.resolve()


def reranker_models_dir() -> Path:
    return DEFAULT_RERANKER_MODELS_DIR.resolve()


def model_label_from_key(model_key: str) -> str:
    return model_key.replace("_", " ").replace("-", " ").title()


def discover_embedding_models(models_dir: Path | None = None) -> list[EmbeddingModelConfig]:
    root = Path(models_dir) if models_dir else embedding_models_dir()
    if not root.exists():
        return []
    models = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_dir() and not path.name.startswith("."):
            models.append(
                EmbeddingModelConfig(
                    key=path.name,
                    label=model_label_from_key(path.name),
                    repo_id=path.name,
                    model_dir=path.resolve(),
                    notes="Local embedding model.",
                )
            )
    return models


def discover_reranker_models(models_dir: Path | None = None) -> list[RerankerModelConfig]:
    root = Path(models_dir) if models_dir else reranker_models_dir()
    if not root.exists():
        return []
    models = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_dir() and not path.name.startswith("."):
            models.append(
                RerankerModelConfig(
                    key=path.name,
                    label=model_label_from_key(path.name),
                    repo_id=path.name,
                    model_dir=path.resolve(),
                )
            )
    return models


def available_embedding_models() -> list[EmbeddingModelConfig]:
    return discover_embedding_models()


def available_reranker_models() -> list[RerankerModelConfig]:
    return discover_reranker_models()


def get_embedding_model(key: str) -> EmbeddingModelConfig:
    for model in available_embedding_models():
        if model.key == key:
            return model
    available = ", ".join(model.key for model in available_embedding_models()) or "none"
    raise ValueError(f"Unknown embedding model key '{key}'. Available keys: {available}")
