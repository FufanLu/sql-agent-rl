# sql-agent-rl

Post-training a multi-turn text-to-SQL agent with reinforcement learning.

The agent gets a database schema and a question. It can run SQL against the
real database through an `execute_sql` tool, read the result, and revise,
for up to 5 tool calls. Then it answers with one final query. Training rewards
only one thing: whether the final query returns the same rows as the gold
query.

Pipeline: data → SFT (verl's SFT trainer) → multi-turn GRPO (verl: Ray +
FSDP + vLLM) → evaluation → serving (vLLM behind the
[vllm-sglang-infra](https://github.com/FufanLu/vllm-sglang-infra) gateway,
plus an MCP server). Planned, not done yet: DPO baseline (TRL), quantization.

> Numbers are added only once they have been measured. Every job is in
> [docs/runs.md](docs/runs.md), every failure in [docs/failures.md](docs/failures.md).

## Architecture

```
                 ┌──────────── verl (Ray single controller) ─────────────┐
 parquet data ──►│  AgentLoop workers ──generate──► vLLM (rollout)        │
 (sar.data)      │       │  ▲                          ▲ weights synced   │
                 │  tool call│ result                   │ every step       │
                 │       ▼  │                    FSDP actor (train)        │
                 │   SqlTool ─────► sql_env ◄──── reward: final SQL vs gold │
                 └──────────────── (read-only SQLite sandbox) ────────────┘
```

| Package | Role |
|---|---|
| `src/sar/sql_env` | Read-only, time- and row-limited SQLite execution; result comparison; FastAPI service with Prometheus metrics |
| `src/sar/agent` | `SqlTool`, the verl tool the agent calls during rollouts |
| `src/sar/reward` | `compute_score`, the verl reward function |
| `src/sar/data` | Prompt and data building, with checks (DB files exist, gold SQL runs, no train/eval overlap) |
| `configs/` | Hydra configs: `grpo/base.yaml` plus one file per run; `tools/sql.yaml` |
| `cluster/` | Slurm jobs for the Explorer cluster: env build, downloads, smoke test, chained GRPO legs |

## Data

- Train: [SkyRL-SQL-653](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data) (540 SynSQL + 113 Spider train questions).
- Eval: Spider dev/test, Spider-DK, Spider-Syn, Spider-Realistic, and BIRD dev (via [SkyRL-SQL-eval](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-eval)), on the databases from [OmniSQL](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets).
- Metric: execution accuracy (EX). The training reward uses set comparison, as SkyRL-SQL and OmniSQL do. A strict variant is also reported: duplicates count, and order counts when the gold query sorts.

## Results

Qwen2.5-Coder-3B-Instruct, one H200 per run, the same 342 training questions
and GRPO settings for every RL run (batch 128 × 5 samples, 60 steps, lr 1e-6,
no KL). Greedy decoding, execution accuracy on all 6,791 eval questions.

| Model | Spider dev | Spider test | Spider-DK | Spider-Syn | Spider-Realistic | BIRD dev | All | Tool calls / question |
|---|---|---|---|---|---|---|---|---|
| Base | 0.077 | 0.121 | 0.028 | 0.082 | 0.089 | 0.059 | 0.085 | 0 |
| RL only | 0.743 | 0.765 | **0.675** | **0.633** | 0.703 | 0.351 | 0.636 | 0 |
| SFT, `<tool_call>` embedding initialized | 0.683 | 0.714 | 0.563 | 0.579 | 0.636 | 0.308 | 0.579 | 1.35–1.79 |
| ↳ then RL | 0.741 | 0.758 | 0.621 | 0.600 | 0.701 | 0.354 | 0.625 | 0 |
| SFT, `<tool_call>` embedding trained first | 0.708 | 0.745 | 0.609 | 0.582 | 0.691 | 0.341 | 0.609 | 1.39–2.13 |
| ↳ then RL | **0.758** | **0.780** | 0.639 | 0.624 | **0.742** | **0.379** | **0.648** | 1.17–1.53 |

What the rows show:

- **RL alone** learns to answer in one turn and never calls the tool. The base
  model scores low mostly because it rarely produces the answer format (6–17%
  of questions answered).
- **Qwen2.5-Coder's `<tool_call>` token is untrained** (its embedding is all
  zeros in the 7B and 32B models; a small placeholder vector in the tied 3B
  model), so SFT cannot teach the model to emit it. Initializing the embedding
  from the mean of its spelling's pieces makes greedy decoding work, but under
  sampling 55% of responses spell the token out as text. RL from that model then
  **collapses to single-turn**: a real tool call carries more ways to fail than
  answering directly, and by step 25 every sample answers directly.
- **Training the two tool-call embedding rows alone first** (lr 1e-3, all
  other weights frozen; p(`<tool_call>`) at the answer start 0.000 → 0.883),
  then SFT, removes the defect: 0 of 640 samples break the token, and RL stays
  multi-turn (verl `num_turns` 4.7 in validation at step 60; 2.0 means no tool call). It is the best
  model overall and on BIRD, with fewer wasted calls than its SFT start
  (questions with 3+ calls 1360 → 265). It is behind RL-only on Spider-DK and
  Spider-Syn.

Ablation, keeping the empty-gold questions (`qwen3b_mt_alldata`): 45% of the
public training set has gold SQL that returns no rows on today's databases, so
any empty result scores there. RL-only on all 621 questions instead of the 342
non-empty ones, everything else the same. On those training questions the
share of empty answers rose from about 0.5 to 0.85 (formatted samples), but it
did not spread: on held-out questions whose gold is not empty, both models
answer empty 3.3% of the time (57 vs 56 questions, sign test p = 1.0), and EX
is level (0.641 vs 0.636, p = 0.15). At this scale the filter is not needed to
protect accuracy (`tools/empty_compare.py`, `docs/runs.md`).

Deployment check (`cluster/deploy_demo.sbatch`, Spider dev + BIRD dev, 2,567
questions): through the gateway, a request without an API key gets 401; 6,187
model requests produced exactly 6,187 usage records for the tenant; final SQL
matched a direct-to-vLLM run on 90.4% of questions, while two direct runs
match on 88.4% (vLLM's run-to-run nondeterminism under greedy decoding), so the
gateway changes nothing; an MCP client answered 5/5 Spider dev questions
through the MCP server and the gateway.

## Environment

The cluster's H200 nodes run NVIDIA driver 570 (CUDA 12.8), while verl 0.9.1's
lock file targets CUDA 13. The env pins **verl 0.9.1 + vLLM 0.26.0 (cu129) +
torch 2.11.0 (cu129) + transformers 5.9.0**; see `cluster/setup_env.sbatch`.

## Develop

```bash
make dev && make check      # CPU-only: lint, mypy, tests
make push                   # copy code to the cluster
```
