from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_CONFIG_PATH = PROJECT_ROOT / "app_config.toml"
DEFAULT_EMBEDDING_MODEL = "voyage-multilingual-2"
APP_CONFIG_SECTIONS = {"snowflake", "columns"}
SNOWFLAKE_CONFIG_KEYS = {"table", "row_limit"}

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
    dimensions: int
    language: str


EMBEDDING_MODELS = [
    EmbeddingModelConfig("snowflake-arctic-embed-l-v2.0", "Snowflake Arctic Embed L v2.0", 1024, "Multilingual"),
    EmbeddingModelConfig("snowflake-arctic-embed-l-v2.0-8k", "Snowflake Arctic Embed L v2.0 8K", 1024, "Multilingual"),
    EmbeddingModelConfig("nv-embed-qa-4", "NV Embed QA 4", 1024, "English"),
    EmbeddingModelConfig("multilingual-e5-large", "Multilingual E5 Large", 1024, "Multilingual"),
    EmbeddingModelConfig("voyage-multilingual-2", "Voyage Multilingual 2", 1024, "Multilingual"),
    EmbeddingModelConfig("snowflake-arctic-embed-m-v1.5", "Snowflake Arctic Embed M v1.5", 768, "English"),
    EmbeddingModelConfig("snowflake-arctic-embed-m", "Snowflake Arctic Embed M", 768, "English"),
    EmbeddingModelConfig("e5-base-v2", "E5 Base v2", 768, "English"),
]


@dataclass(frozen=True)
class ColumnConfig:
    claim_id: str
    description: str
    line_of_business: str = ""
    claim_type: str = ""
    cause_of_loss: str = ""
    damaged_object: str = ""
    country: str = ""
    claim_status: str = ""
    loss_date: str = ""
    reserve_amount: str = ""
    paid_amount: str = ""
    currency: str = ""
    policy_type: str = ""

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ColumnConfig":
        expected = {config_key: attr for _, config_key, attr, _ in COLUMN_MAPPING_FIELDS}
        unknown = sorted(set(values) - set(expected))
        if unknown:
            raise ValueError(
                f"Unknown column mapping(s): {', '.join(f'columns.{key}' for key in unknown)}"
            )
        missing = [config_key for _, config_key, _, _ in COLUMN_MAPPING_FIELDS if config_key not in values]
        if missing:
            raise ValueError(
                "Missing column mapping(s): "
                f"{', '.join(f'columns.{key}' for key in missing)}. "
                'Set optional mappings to "" to skip them.'
            )
        return cls(
            **{
                attr: str(values[config_key]).strip()
                for _, config_key, attr, _ in COLUMN_MAPPING_FIELDS
            }
        )

    def validate_required(self) -> None:
        missing = [
            config_key
            for _, config_key, attr, required in COLUMN_MAPPING_FIELDS
            if required and not getattr(self, attr)
        ]
        if missing:
            raise ValueError(f"Required column mapping(s) cannot be blank: {', '.join(f'columns.{key}' for key in missing)}")

    @property
    def selected_columns(self) -> list[str]:
        seen: set[str] = set()
        ordered = [getattr(self, attr) for _, _, attr, _ in COLUMN_MAPPING_FIELDS]
        return [column for column in ordered if column and not (column in seen or seen.add(column))]

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
    columns: ColumnConfig
    snowflake_table: str = ""
    snowflake_row_limit: int | None = None

    @classmethod
    def from_app_config(cls) -> "AppConfig":
        if not DEFAULT_APP_CONFIG_PATH.exists():
            raise RuntimeError(f"App config file was not found: {DEFAULT_APP_CONFIG_PATH}")
        with DEFAULT_APP_CONFIG_PATH.open("rb") as file:
            app_data = tomllib.load(file)
        unknown_sections = sorted(set(app_data) - APP_CONFIG_SECTIONS)
        if unknown_sections:
            raise ValueError(f"Unknown app config section(s): {', '.join(unknown_sections)}")
        if "snowflake" not in app_data or not isinstance(app_data["snowflake"], dict):
            raise ValueError("Required app config section [snowflake] is missing or invalid")
        if "columns" not in app_data or not isinstance(app_data["columns"], dict):
            raise ValueError("Required app config section [columns] is missing or invalid")

        snowflake = app_data["snowflake"]
        unknown_snowflake_keys = sorted(set(snowflake) - SNOWFLAKE_CONFIG_KEYS)
        if unknown_snowflake_keys:
            raise ValueError(
                f"Unknown Snowflake config value(s): {', '.join(f'snowflake.{key}' for key in unknown_snowflake_keys)}"
            )
        if "table" not in snowflake:
            raise ValueError("Required app config value is missing: snowflake.table")
        row_limit = None
        if snowflake.get("row_limit") not in (None, ""):
            row_limit = int(snowflake["row_limit"])
        return cls(
            snowflake_table=str(snowflake["table"]).strip(),
            snowflake_row_limit=row_limit,
            columns=ColumnConfig.from_mapping(app_data["columns"]),
        )

    def validate_source(self) -> None:
        if not self.snowflake_table:
            raise ValueError("Required app config value cannot be blank: snowflake.table")
        if self.snowflake_row_limit is not None and self.snowflake_row_limit <= 0:
            raise ValueError("snowflake.row_limit must be a positive integer when set")
        self.columns.validate_required()

    @property
    def embedding_table(self) -> str:
        prefix, separator, table = self.snowflake_table.rpartition(".")
        if table.startswith('"') and table.endswith('"'):
            table = f'"{table[1:-1]}_EMBEDDINGS"'
        else:
            table = f"{table}_EMBEDDINGS"
        return f"{prefix}{separator}{table}"


def available_embedding_models() -> list[EmbeddingModelConfig]:
    return EMBEDDING_MODELS.copy()


def get_embedding_model(key: str) -> EmbeddingModelConfig:
    for model in EMBEDDING_MODELS:
        if model.key == key:
            return model
    raise ValueError(f"Unknown Snowflake embedding model: {key}")


def embedding_column_for_model(model_key: str) -> str:
    get_embedding_model(model_key)
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", model_key).strip("_").upper()
    return f"EMBEDDING_{suffix}"


def model_for_embedding_column(column_name: str) -> EmbeddingModelConfig | None:
    normalized = column_name.strip('"').upper()
    return next(
        (model for model in EMBEDDING_MODELS if embedding_column_for_model(model.key) == normalized),
        None,
    )


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
