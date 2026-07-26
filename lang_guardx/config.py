from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _import_tomllib():
    """Lazy import of TOML parser (Python 3.11+ stdlib or fallback to tomli)."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError("TOML support requires Python 3.11+ or the 'tomli' package") from exc
    return tomllib


def _import_tomli_w():
    """Lazy import of TOML writer."""
    try:
        import tomli_w  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("TOML write support requires the 'tomli-w' package") from exc
    return tomli_w


# ── Detection sub-configs ──────────────────────────────────────────────────────


class BloomConfig(BaseModel):
    """Configuration for the Bloom-filter detection layer."""

    enabled: bool = True
    capacity: int = 100_000
    false_positive_rate: float = 0.001
    corpus_path: str | None = None


class RegexConfig(BaseModel):
    """Configuration for the regex detection layer."""

    enabled: bool = True


class DistilBertConfig(BaseModel):
    """Configuration for the DistilBERT neural detection layer."""

    enabled: bool = True
    model_path: str | None = None
    threshold: float = 0.75
    brm_cost_fp: float = 1.0
    brm_cost_fn: float = 2.0
    brm_uncertain_ratio: float = 0.2


class DetectionConfig(BaseModel):
    """Configuration for Layer 1 — input detection."""

    enabled: bool = True
    bloom: BloomConfig = Field(default_factory=BloomConfig)
    regex: RegexConfig = Field(default_factory=RegexConfig)
    distilbert: DistilBertConfig = Field(default_factory=DistilBertConfig)


# ── Policy sub-configs ─────────────────────────────────────────────────────────


class PolicyConfig(BaseModel):
    """Configuration for the SQL security policy."""

    permitted_operations: list[str] = ["SELECT"]
    permitted_tables: list[str] = Field(default_factory=list)
    restricted_columns: dict[str, list[str]] = Field(default_factory=dict)
    scoped_tables: list[str] = Field(default_factory=list)
    require_user_scope: bool = False
    max_rows: int = 1000


class EngineConfig(BaseModel):
    """Configuration for the SQL policy engine (Layer 2)."""

    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    dialect: str = "sqlite"
    current_user_id: int | None = None


# ── Adaptive sub-configs ───────────────────────────────────────────────────────


class AdaptiveConfig(BaseModel):
    """Configuration for Layer 4 — runtime adaptation."""

    enabled: bool = True
    log_path: str = "adaptive_log.jsonl"
    taxonomy_path: str | None = None


# ── LangChain integration ──────────────────────────────────────────────────────


class LangChainConfig(BaseModel):
    """Configuration for optional LangChain agent integration."""

    enabled: bool = False
    top_k: int = 10


# ── Top-level Config ───────────────────────────────────────────────────────────


class Config(BaseModel):
    """Central configuration for LangGuardX.

    Can be loaded from YAML, JSON, TOML, a dict, or environment variables.
    Every field has a sensible default so ``Config()`` gives you a working setup.
    """

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load configuration from a TOML file."""
        tomllib = _import_tomllib()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Load configuration from a plain dict."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Recognised variables:
          LANGGUARDX_DETECTION_ENABLED
          LANGGUARDX_ADAPTIVE_ENABLED
          LANGGUARDX_POLICY_PERMITTED_OPS
          LANGGUARDX_POLICY_PERMITTED_TABLES
          LANGGUARDX_POLICY_MAX_ROWS
          LANGGUARDX_DIALECT
          LANGGUARDX_LANGCHAIN_ENABLED
          LANGGUARDX_LANGCHAIN_TOP_K
        """
        cfg = cls()

        if (v := os.getenv("LANGGUARDX_DETECTION_ENABLED")) is not None:
            cfg.detection.enabled = v.lower() == "true"

        if (v := os.getenv("LANGGUARDX_ADAPTIVE_ENABLED")) is not None:
            cfg.adaptive.enabled = v.lower() == "true"

        if (v := os.getenv("LANGGUARDX_POLICY_PERMITTED_OPS")) is not None:
            cfg.engine.policy.permitted_operations = [x.strip() for x in v.split(",") if x.strip()]

        if (v := os.getenv("LANGGUARDX_POLICY_PERMITTED_TABLES")) is not None:
            cfg.engine.policy.permitted_tables = [x.strip() for x in v.split(",") if x.strip()]

        if (v := os.getenv("LANGGUARDX_POLICY_MAX_ROWS")) is not None:
            cfg.engine.policy.max_rows = int(v)

        if (v := os.getenv("LANGGUARDX_DIALECT")) is not None:
            cfg.engine.dialect = v

        if (v := os.getenv("LANGGUARDX_LANGCHAIN_ENABLED")) is not None:
            cfg.langchain.enabled = v.lower() == "true"

        if (v := os.getenv("LANGGUARDX_LANGCHAIN_TOP_K")) is not None:
            cfg.langchain.top_k = int(v)

        return cfg

    def save(self, path: str | Path, format: str | None = None) -> None:
        """Save configuration to a file.

        The output format is inferred from the file suffix when *format* is ``None``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        fmt = format or {"yaml": "yaml", "yml": "yaml", "json": "json", "toml": "toml"}.get(suffix.lstrip("."), "yaml")
        data = self.model_dump(mode="python")
        if fmt == "yaml":
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif fmt == "toml":
            tomli_w = _import_tomli_w()
            with open(path, "wb") as f:
                tomli_w.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
