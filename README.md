# CSRA — Customer Support Resolution Agent

**CSRA (Customer Support Resolution Agent)** is an agentic AI customer support portfolio project designed to understand customer issues, investigate them using tools, retrieve relevant policies, propose resolutions, and execute permitted actions under controlled conditions.

## Current Status

- **Phase 0 (Complete)**: Project scaffolding, centralized configuration (`src/csra/utils/config.py`), logging, and minimal FastAPI app (`src/csra/main.py`).
- **Phase 1 (Complete)**: Capability-based **Cloudflare Workers AI** LLM evaluation pipeline (`evals/`) benchmarking models across 8 core CSRA capabilities using 40 realistic cases.
- **Phase 2 (Next)**: LangGraph agent workflow, simulated tools, and policy retrieval.

## Quick Start

1. **Install dependencies** (Python 3.12+):

   ```bash
   pip install -e ".[dev]"
   ```

2. **Configure Cloudflare Workers AI** in `.env`:

   ```env
   CLOUDFLARE_API_KEY=your-cloudflare-api-token
   CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
   ```

3. **Run tests**:

   ```bash
   pytest
   ```

4. **Run capability evaluation**:

   ```bash
   # Smoke test (8 cases)
   python -m evals.run_evaluation --smoke

   # Full evaluation (40 cases)
   python -m evals.run_evaluation

   # Inspect failures or history
   python -m evals.run_evaluation --show-failures
   python -m evals.run_evaluation --history
   ```

5. **Run the FastAPI server**:

   ```bash
   uvicorn csra.main:app --reload
   ```
