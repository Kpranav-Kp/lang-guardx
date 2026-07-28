from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, field_validator


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

    @field_validator("false_positive_rate")
    @classmethod
    def _valid_fpr(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("false_positive_rate must be in (0, 1)")
        return v

    @field_validator("capacity")
    @classmethod
    def _positive_capacity(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("capacity must be positive")
        return v


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

    @field_validator("threshold")
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return v

    @field_validator("brm_cost_fp", "brm_cost_fn")
    @classmethod
    def _positive_cost(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("brm_cost_fp and brm_cost_fn must be positive")
        return v

    @field_validator("brm_uncertain_ratio")
    @classmethod
    def _valid_uncertain_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("brm_uncertain_ratio must be in [0, 1]")
        return v


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

    @field_validator("permitted_operations")
    @classmethod
    def _non_empty_ops(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("permitted_operations cannot be empty")
        return [op.upper() for op in v]

    @field_validator("max_rows")
    @classmethod
    def _valid_max_rows(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_rows must be non-negative")
        return v


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

    @field_validator("top_k")
    @classmethod
    def _positive_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


# ── Concurrency sub-config ─────────────────────────────────────────────────────


class ConcurrencyConfig(BaseModel):
    """Configuration for thread safety and async operation."""

    enabled: bool = True
    max_workers: int = 4
    async_timeout: float = 30.0


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
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)

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

        Uses ``LANGGUARDX_`` prefix with ``__`` as the path separator::

          LANGGUARDX_DETECTION__BLOOM__ENABLED = false
          LANGGUARDX_ENGINE__POLICY__PERMITTED_TABLES = products, orders
          LANGGUARDX_DETECTION__DISTILBERT__THRESHOLD = 0.85
          LANGGUARDX_ADAPTIVE__ENABLED = false

        Supported value types:
          * ``true``/``false``, ``1``/``0``, ``yes``/``no``  →  ``bool``
          * comma-separated items                                →  ``list[str]``
          * ``key=val,key=val``                                   →  ``dict[str, str]``
          * numeric strings                                       →  ``int`` / ``float``
        """
        cfg = cls()
        for key, value in sorted(os.environ.items()):
            if not key.startswith("LANGGUARDX_") or key == "LANGGUARDX":
                continue
            suffix = key[len("LANGGUARDX_") :]
            parts = suffix.lower().split("__")

            # Walk the nested model tree
            obj = cfg
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                field_name = parts[-1]
                if hasattr(obj, field_name):
                    ann = obj.__class__.model_fields[field_name].annotation
                    typed_value = cls._coerce_env(value, ann) if ann is not None else value
                    setattr(obj, field_name, typed_value)
        # Re-validate so @field_validator decorators run on env-supplied values
        return cls.model_validate(cfg.model_dump())

    @staticmethod
    def _coerce_env(value: str, ann: Any) -> Any:
        """Convert an env-var string to the target Python type according to *ann*."""
        # Unwrap Optional[X] → X
        origin = get_origin(ann)
        args = get_args(ann)
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                ann = non_none[0]
                origin = get_origin(ann)
                args = get_args(ann)

        # bool
        if ann is bool:
            return value.lower() in ("true", "1", "yes")
        # int
        if ann is int:
            return int(value)
        # float
        if ann is float:
            return float(value)
        # list[X]
        if origin is list and args:
            item_type = args[0]
            items = [x.strip() for x in value.split(",") if x.strip()]
            if item_type is str:
                return items
            return [item_type(x) for x in items]
        # dict[K, V]  —  expects "k=v,k=v" format
        if origin is dict and args:
            k_type, v_type = args
            result: dict = {}
            for pair in value.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k_type is not str:
                        k = k_type(k)
                    if v_type is not str:
                        v = v_type(v)
                    result[k] = v
            return result
        # str (default)
        return value

    def save(self, path: str | Path, fmt: str | None = None) -> None:
        """Save configuration to a file.

        The output format is inferred from the file suffix when *fmt* is ``None``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        fmt = fmt or {"yaml": "yaml", "yml": "yaml", "json": "json", "toml": "toml"}.get(suffix.lstrip("."), "yaml")
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
