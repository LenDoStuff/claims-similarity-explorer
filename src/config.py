from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_EMBEDDING_MODELS_DIR = DEFAULT_MODELS_DIR / "embeddings"
DEFAULT_RERANKER_MODELS_DIR = DEFAULT_MODELS_DIR / "rerankers"
DEFAULT_MODEL_DIR = DEFAULT_EMBEDDING_MODELS_DIR / "multilingual-e5-small"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_COLLECTION_NAME = "claims"
DEFAULT_MODEL_KEY = "multilingual-e5-small"
DEFAULT_EMBEDDING_MODEL_NAME = DEFAULT_MODEL_KEY
DEFAULT_EMBEDDING_VERSION = "e5-v1"
DEFAULT_RERANKER_MODEL_NAME = "mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_MODEL_DIR = DEFAULT_RERANKER_MODELS_DIR / DEFAULT_RERANKER_MODEL_NAME


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
    def from_env(cls) -> "ColumnConfig":
        return cls(
            claim_id=os.getenv("CLAIM_ID_COLUMN", cls.claim_id),
            description=os.getenv("CLAIM_DESCRIPTION_COLUMN", cls.description),
            line_of_business=os.getenv("LINE_OF_BUSINESS_COLUMN", cls.line_of_business),
            claim_type=os.getenv("CLAIM_TYPE_COLUMN", cls.claim_type),
            cause_of_loss=os.getenv("CAUSE_OF_LOSS_COLUMN", cls.cause_of_loss),
            damaged_object=os.getenv("DAMAGED_OBJECT_COLUMN", cls.damaged_object),
            country=os.getenv("COUNTRY_COLUMN", cls.country),
            claim_status=os.getenv("CLAIM_STATUS_COLUMN", cls.claim_status),
            loss_date=os.getenv("LOSS_DATE_COLUMN", cls.loss_date),
            reserve_amount=os.getenv("RESERVE_AMOUNT_COLUMN", cls.reserve_amount),
            paid_amount=os.getenv("PAID_AMOUNT_COLUMN", cls.paid_amount),
            currency=os.getenv("CURRENCY_COLUMN", cls.currency),
            policy_type=os.getenv("POLICY_TYPE_COLUMN", cls.policy_type),
        )

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
        return [
            self.description,
            self.line_of_business,
            self.claim_type,
            self.cause_of_loss,
            self.damaged_object,
            self.country,
        ]


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    table: str
    role: str | None = None

    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        required = {
            "SNOWFLAKE_ACCOUNT": os.getenv("SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_USER": os.getenv("SNOWFLAKE_USER"),
            "SNOWFLAKE_PASSWORD": os.getenv("SNOWFLAKE_PASSWORD"),
            "SNOWFLAKE_WAREHOUSE": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "SNOWFLAKE_DATABASE": os.getenv("SNOWFLAKE_DATABASE"),
            "SNOWFLAKE_SCHEMA": os.getenv("SNOWFLAKE_SCHEMA"),
            "SNOWFLAKE_TABLE": os.getenv("SNOWFLAKE_TABLE"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required Snowflake environment variable(s): {joined}")

        return cls(
            account=required["SNOWFLAKE_ACCOUNT"] or "",
            user=required["SNOWFLAKE_USER"] or "",
            password=required["SNOWFLAKE_PASSWORD"] or "",
            warehouse=required["SNOWFLAKE_WAREHOUSE"] or "",
            database=required["SNOWFLAKE_DATABASE"] or "",
            schema=required["SNOWFLAKE_SCHEMA"] or "",
            table=required["SNOWFLAKE_TABLE"] or "",
            role=os.getenv("SNOWFLAKE_ROLE") or None,
        )

    @property
    def qualified_table(self) -> str:
        return ".".join(quote_identifier(part) for part in [self.database, self.schema, self.table])


@dataclass(frozen=True)
class AppConfig:
    model_key: str = DEFAULT_MODEL_KEY
    columns: ColumnConfig = field(default_factory=ColumnConfig.from_env)
    model_dir: Path = DEFAULT_MODEL_DIR
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_version: str = DEFAULT_EMBEDDING_VERSION

    @classmethod
    def from_env(cls) -> "AppConfig":
        model_key = os.getenv("EMBEDDING_MODEL_KEY")
        model = get_embedding_model(model_key) if model_key else get_default_embedding_model()
        return cls(
            model_key=model.key if model else DEFAULT_MODEL_KEY,
            columns=ColumnConfig.from_env(),
            model_dir=(
                model.model_dir
                if model
                else Path(os.getenv("EMBEDDING_MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()
            ),
            chroma_dir=Path(os.getenv("CHROMA_DB_DIR", str(DEFAULT_CHROMA_DIR))).resolve(),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR))).resolve(),
            collection_name=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME),
            embedding_model_name=(
                model.repo_id if model else os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)
            ),
            embedding_version=os.getenv("EMBEDDING_VERSION", DEFAULT_EMBEDDING_VERSION),
        )

    @property
    def index_manifest_path(self) -> Path:
        return self.artifacts_dir / "index_manifest.json"

    @property
    def clusters_path(self) -> Path:
        return self.artifacts_dir / "clusters.json"

    @property
    def cluster_map_path(self) -> Path:
        return self.artifacts_dir / "cluster_map.json"

    def collection_name_for_model(self, model_key: str) -> str:
        return model_collection_name(self.collection_name, model_key)

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


def get_config() -> AppConfig:
    return AppConfig.from_env()


def model_storage_key(model_key: str) -> str:
    storage_key = re.sub(r"[^0-9A-Za-z_-]+", "_", model_key).strip("_")
    return storage_key.replace("-", "_").lower()


def model_collection_name(base_collection_name: str, model_key: str) -> str:
    return f"{base_collection_name}_{model_storage_key(model_key)}"


def versioned_collection_name(base_collection_name: str, model_key: str, index_hash: str) -> str:
    return f"{model_collection_name(base_collection_name, model_key)}_{index_hash[:12]}"


def models_root() -> Path:
    return Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).resolve()


def embedding_models_dir() -> Path:
    return Path(os.getenv("EMBEDDING_MODELS_DIR", str(models_root() / "embeddings"))).resolve()


def reranker_models_dir() -> Path:
    return Path(os.getenv("RERANKER_MODELS_DIR", str(models_root() / "rerankers"))).resolve()


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


def get_default_embedding_model() -> EmbeddingModelConfig | None:
    models = available_embedding_models()
    if not models:
        return None
    return next((model for model in models if model.key == DEFAULT_MODEL_KEY), models[0])


def get_reranker_model(key: str | None = None) -> RerankerModelConfig:
    models = available_reranker_models()
    if key is None:
        if not models:
            raise ValueError("No local reranker models found.")
        return models[0]
    for model in models:
        if model.key == key:
            return model
    available = ", ".join(model.key for model in models) or "none"
    raise ValueError(f"Unknown reranker model key '{key}'. Available keys: {available}")
