# sql-agent-rl

Post-training a multi-turn text-to-SQL agent with reinforcement learning.

The agent gets a database schema and a question. It can run SQL against the
real database through an `execute_sql` tool, read the result, and revise,
for up to 5 tool calls. Then it answers with one final query. Training rewards
only one thing: whether the final query returns the same rows as the gold
query.

Pipeline: data → SFT (LLaMA-Factory) → DPO baseline (TRL) → multi-turn GRPO
(verl: Ray + FSDP + vLLM) → evaluation → quantization → serving (vLLM behind
the [vllm-sglang-infra](https://github.com/FufanLu/vllm-sglang-infra) gateway,
plus an MCP server).

> Status: under construction. Numbers are added only once they have been
> measured. Failures are logged in [docs/failures.md](docs/failures.md).

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

## Environment

The cluster's H200 nodes run NVIDIA driver 570 (CUDA 12.8), while verl 0.9.1's
lock file targets CUDA 13. The env pins **verl 0.9.1 + vLLM 0.26.0 (cu129) +
torch 2.11.0 (cu129) + transformers 5.9.0**; see `cluster/setup_env.sbatch`.

## Develop

```bash
make dev && make check      # CPU-only: lint, mypy, tests
make push                   # copy code to the cluster
```
