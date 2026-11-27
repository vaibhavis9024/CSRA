"""Cloudflare Workers AI OpenAI-compatible adapter."""

from __future__ import annotations

from evals.models.adapters.openai_compatible import OpenAICompatibleAdapter


def build_cloudflare_base_url(account_id: str) -> str:
    """Build the OpenAI-compatible Workers AI base URL for a Cloudflare account."""
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id.strip()}/ai/v1"


class CloudflareAdapter(OpenAICompatibleAdapter):
    """Adapter for Cloudflare Workers AI's OpenAI-compatible /ai/v1 endpoint."""

    provider_name = "cloudflare"
