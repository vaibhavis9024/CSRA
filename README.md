# CSRA — Customer Support Resolution Agent

**CSRA (Customer Support Resolution Agent)** is an agentic AI customer support portfolio project designed to understand customer issues, investigate them using tools, retrieve relevant policies, propose resolutions, and execute permitted actions under controlled conditions.

This repository currently contains:
- **Phase 0 (`src/csra/`)**: Project scaffolding, centralized settings, logging, and minimal FastAPI app.
- **Phase 1 (`evals/`)**: A lightweight, standalone **Cloudflare Workers AI** capability evaluation pipeline used to select the best LLM for CSRA before building the agent.

---

## Evaluated Capabilities (8)

1. **Intent Understanding** *(deterministic)* — Primary/multiple intents and ambiguity detection.
2. **Information Extraction** *(deterministic)* — Order IDs, products, amounts, dates, and self-corrections without hallucination.
3. **Tool Selection & Calling** *(deterministic)* — Choosing the right simulated tool(s), valid arguments, and abstaining when info is missing.
4. **Multi-step Reasoning** *(LLM judge)* — Combining policy/context evidence to reach valid support decisions.
5. **Customer Communication** *(LLM judge)* — Clarity, empathy, professionalism, and avoiding false promises.
6. **Knowledge & Policy Retrieval** *(LLM judge)* — Grounding answers in provided policy snippets and admitting missing info.
7. **Context Handling** *(deterministic)* — Retaining IDs, resolving references, and handling corrections across turns.
8. **Safety & Policy Compliance** *(deterministic)* — Refusing prompt injections, policy bypasses, PII leaks, and escalating security incidents.

---

## Candidate Models & Judge

Configured in `configs/evaluation.yaml` (all accessed via Cloudflare Workers AI's free-tier OpenAI-compatible endpoint):

- **Candidate Models**:
  - `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (`Llama 3.3 70B`)
  - `@cf/mistralai/mistral-small-3.1-24b-instruct` (`Mistral Small 3.1`)
  - `@cf/google/gemma-4-26b-a4b-it` (`Gemma 4 26B`)
  - `@cf/qwen/qwen3.8-27b` (`Qwen 3.8 27B`)
  - `@cf/zai-org/glm-4.7-flash` (`GLM 4.7 Flash`)
- **Judge Model**: `@cf/ibm-granite/granite-4.0-h-micro` (kept separate from candidate models so self-judging is never triggered).

---

## Methodology & Scoring

- **Dataset**: `data/evals/capability_cases.jsonl` (40 realistic cases, 5 per capability).
- **0–2 Per-Case Scale**: `0` = Fail, `1` = Partial, `2` = Pass.
- **Capability Score (%)**: `(Total points earned / Maximum possible points) * 100`.
- **Safety First**: Safety failures are counted and displayed separately—a high reference average never hides a safety failure.

---

## Setup & Execution

1. **Install dependencies** (Python 3.12+):

   ```bash
   pip install -e ".[dev]"
   ```

2. **Configure Cloudflare Workers AI** in `.env`:

   ```env
   CLOUDFLARE_API_KEY=your-cloudflare-api-token
   CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
   ```

3. **Run unit tests** (mocked, no API calls):

   ```bash
   pytest
   ```

4. **Run evaluations**:

   ```bash
   # Quick smoke test (1 case per capability = 8 cases)
   python -m evals.run_evaluation --smoke

   # Full evaluation (all 40 cases across enabled models)
   python -m evals.run_evaluation

   # Inspect failures or historical runs
   python -m evals.run_evaluation --show-failures
   python -m evals.run_evaluation --show-failures --detailed
   python -m evals.run_evaluation --history
   ```

---

## Output & Results Interpretation

- The terminal displays a side-by-side capability percentage table, reference average, average latency, safety failures, and total failed cases (unavailable metrics render as `N/A`).
- Results are saved under `data/evals/results/`:
  - `latest.txt` — Most recent comparison table
  - `history.csv` — One row per model per run
  - `runs/<run_id>/summary.txt` and `case_results.jsonl`

## Limitations

- The 40-case dataset is synthetic and requires human review before treating scores as final ground truth.
- Subjective capabilities rely on an LLM judge, which may introduce scoring variance.
