"""Simple Cloudflare Workers AI model registry and evaluation configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from csra.utils.config import Settings
from evals.models.cloudflare import CloudflareAdapter, build_cloudflare_base_url


class ModelEntry(BaseModel):
    """A single candidate model in configs/evaluation.yaml."""

    id: str
    name: str
    enabled: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


class JudgeConfig(BaseModel):
    """Optional LLM judge configuration."""

    enabled: bool = True
    model_id: str = "@cf/ibm-granite/granite-4.0-h-micro"


class EvaluationConfig(BaseModel):
    """Top-level evaluation configuration loaded from configs/evaluation.yaml."""

    dataset_path: str = "data/evals/capability_cases.jsonl"
    dataset_version: str = "csra_capability_v1"
    results_dir: str = "data/evals/results"
    concurrency: int = 2
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.0
    max_tokens: int = 1024
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    models: list[ModelEntry] = Field(default_factory=list)


def load_evaluation_config(config_path: str | Path = "configs/evaluation.yaml") -> EvaluationConfig:
    """Load and validate evaluation configuration from YAML."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return EvaluationConfig.model_validate(raw)


class ModelRegistry:
    """Resolves configured models into CloudflareAdapter instances."""

    def __init__(self, config: EvaluationConfig, settings: Settings | None = None) -> None:
        self.config = config
        self.settings = settings or Settings()

    def enabled_models(self) -> list[ModelEntry]:
        """Return all enabled model entries."""
        return [m for m in self.config.models if m.enabled]

    def get_model(self, name_or_id: str) -> ModelEntry:
        """Find a model entry by display name or Cloudflare model ID."""
        for m in self.config.models:
            if m.name == name_or_id or m.id == name_or_id:
                return m
        available = ", ".join(m.name for m in self.config.models)
        raise KeyError(f"Unknown model '{name_or_id}'. Available: {available}")

    def _resolve_api_key(self, require_credentials: bool) -> str | None:
        env_key = os.getenv("CLOUDFLARE_API_KEY") or os.getenv("CLOUDFLARE_API_TOKEN")
        secret_key = (
            self.settings.cloudflare_api_key.get_secret_value()
            if self.settings.cloudflare_api_key
            else None
        )
        api_key = env_key or secret_key
        if require_credentials and not api_key:
            raise ValueError(
                "Missing Cloudflare API key: set CLOUDFLARE_API_KEY in your environment or .env file."
            )
        return api_key

    def _resolve_base_url(self, require_credentials: bool) -> str:
        explicit_url = os.getenv("CLOUDFLARE_BASE_URL") or self.settings.cloudflare_base_url
        if explicit_url:
            return explicit_url.rstrip("/")

        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID") or self.settings.cloudflare_account_id
        if account_id:
            return build_cloudflare_base_url(account_id)

        if require_credentials:
            raise ValueError(
                "Missing Cloudflare Account ID: set CLOUDFLARE_ACCOUNT_ID "
                "(or CLOUDFLARE_BASE_URL) in your environment or .env file."
            )
        return build_cloudflare_base_url("dry-run-account")

    def create_adapter(
        self,
        model_id: str,
        model_name: str | None = None,
        require_api_key: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CloudflareAdapter:
        """Instantiate a CloudflareAdapter for the given model ID."""
        api_key = self._resolve_api_key(require_credentials=require_api_key)
        base_url = self._resolve_base_url(require_credentials=require_api_key)
        return CloudflareAdapter(
            model_name=model_name or model_id,
            model_id=model_id,
            api_key=api_key,
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            retry_backoff_seconds=self.config.retry_backoff_seconds,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
            base_url=base_url,
        )
