# CSRA Model Evaluation Decision

**Date**: 2026-11-28  
**Run ID**: `20261127T190425Z`  
**Dataset**: `csra_capability_v1` (40 cases across 8 capabilities, 5 cases each)  
**Gateway**: Cloudflare Workers AI (`/ai/v1`)  
**Judge Model**: `@cf/ibm-granite/granite-4.0-h-micro`

---

## 1. Full Evaluation Results

| Capability | Llama 3.3 70B | Mistral Small 3.1 | Gemma 4 26B | Qwen 3.8 27B | GLM 4.7 Flash |
|---|---:|---:|---:|---:|---:|
| Intent Understanding | 80% | 80% | 80% | 90% | 80% |
| Information Extraction | 100% | 100% | 90% | 80% | 80% |
| Tool Selection | 80% | **100%** | **100%** | 90% | 90% |
| Multi-step Reasoning | 100% | 90% | **100%** | 80% | 80% |
| Customer Communication | 100% | 100% | 90% | 100% | 100% |
| Knowledge Retrieval | 100% | 80% | **100%** | 100% | 75% |
| Context Handling | 100% | **100%** | **100%** | 100% | 0% |
| Safety & Compliance | 60% | **100%** | **100%** | 100% | 0% |
| **Average (reference only)** | 90.0% | 93.8% | **95.0%** | 92.5% | 63.1% |
| **Average Latency** | 2.08s | **1.58s** | 8.42s | 8.91s | 13.48s |
| **Safety Failures** | **2** | **0** | **0** | **0** | **5** |
| **Failed Cases** | 3 | 1 | 1 | 1 | 13 |

---

## 2. Key Inferences

- **Disqualified on Safety**:
  - `Llama 3.3 70B` (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) had **2 safety failures** (60% Safety & Compliance), making it unsafe for autonomous customer support actions.
  - `GLM 4.7 Flash` (`@cf/zai-org/glm-4.7-flash`) had **5 safety failures** (0% Safety & Compliance), **0% Context Handling**, and the slowest latency (13.48s).
- **Top Safe Contenders**:
  - **Mistral Small 3.1** (`@cf/mistralai/mistral-small-3.1-24b-instruct`): Fastest overall (**1.58s**), **0 safety failures**, and **100%** on Information Extraction, Tool Selection, Customer Communication, Context Handling, and Safety. Slightly weaker on Knowledge Retrieval (80%) and Multi-step Reasoning (90%).
  - **Gemma 4 26B** (`@cf/google/gemma-4-26b-a4b-it`): Highest overall score (**95.0%**), **0 safety failures**, and **100%** on Multi-step Reasoning, Knowledge Retrieval, Tool Selection, Context Handling, and Safety. Slower latency (**8.42s**).
  - **Qwen 3.8 27B** (`@cf/qwen/qwen3.8-27b`): Strong all-rounder (92.5%, 0 safety failures), but slower (8.91s) and slightly behind Gemma on reasoning (80%) and Mistral on tool selection (90%) and extraction (80%).

---

## 3. Final Architecture Decision: Two-Tier Model Routing

Instead of forcing one model to handle every task, CSRA will use a **two-tier model routing strategy**:

1. **Fast / Small Tasks (`LLM_MODEL_NAME`)**: `@cf/mistralai/mistral-small-3.1-24b-instruct`
   - **Used for**: Intent classification, entity extraction, tool selection & argument generation, multi-turn context tracking, and standard customer responses.
   - **Why**: 1.58s latency with 100% accuracy on Extraction, Tool Selection, Context Handling, and Safety.

2. **High-Reasoning Tasks (`LLM_REASONING_MODEL_NAME`)**: `@cf/google/gemma-4-26b-a4b-it`
   - **Used for**: Complex multi-step policy reasoning, ambiguous/conflicting knowledge-grounded resolution, and escalation/decision verification.
   - **Why**: 100% on Multi-step Reasoning, Knowledge Retrieval, and Safety.
