# CSRA Capability Model Evaluation

A lightweight evaluation pipeline to compare LLMs via **Cloudflare Workers AI** across the eight capabilities required by **CSRA (Customer Support Resolution Agent)**.

## The 8 Capabilities Evaluated

1. **Intent Understanding** (deterministic)
2. **Information Extraction** (deterministic)
3. **Tool Selection & Calling** (deterministic)
4. **Multi-step Reasoning** (LLM judge)
5. **Customer Communication** (LLM judge)
6. **Knowledge & Policy Retrieval** (LLM judge)
7. **Context Handling** (deterministic)
8. **Safety & Policy Compliance** (deterministic)

## Cloudflare Workers AI Setup

Set your Cloudflare API token and Account ID in `.env`:

```env
CLOUDFLARE_API_KEY=your-cloudflare-api-token
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
```

Requests are routed to `https://api.cloudflare.com/client/v4/accounts/<CLOUDFLARE_ACCOUNT_ID>/ai/v1/chat/completions`.

## Adding Models

Edit `configs/evaluation.yaml` and add an entry under `models:` with its Workers AI `@cf/...` model ID:

```yaml
  - id: "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    name: Llama 3.3 70B
    enabled: true
```

## Running Evaluations

```bash
# Quick smoke test (1 case per capability = 8 cases)
python -m evals.run_evaluation --smoke

# Full evaluation (all 40 cases across enabled models)
python -m evals.run_evaluation

# Inspect failed cases or historical runs
python -m evals.run_evaluation --show-failures
python -m evals.run_evaluation --show-failures --detailed
python -m evals.run_evaluation --history
```
