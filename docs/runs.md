# Run log

Every cluster job, successful or not, in submission order. Start, end and
node come from Slurm (`sacct`); since 2026-10-22 22:30 every job also records
itself in `$SAR_ROOT/runs.tsv` when it starts (`cluster/common.sh`), and
`tools/runs_table.py` joins the two. Failures are explained in
[failures.md](failures.md) (F#). Times are cluster time (US Eastern).

Results are copied from the job's own output files; nothing here is estimated.

## 2026-10-22

| Job | What | Partition / node | State | Elapsed | Result |
|---|---|---|---|---|---|
| 11310774 | Build flash-attn 2.8.3 wheel for torch 2.11 + CUDA 12 (sm_80, sm_90) | short / d0146 | completed | ~3h | `wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` |
| 11310779 | Build training env (verl 0.9.1, vLLM 0.26.0 cu129, torch 2.11.0, LLaMA-Factory 0.9.5, TRL, DeepSpeed) | short / c0637 | completed | 12m43s | All packages import together; env tar 12.6 GB. F#2 |
| 11310781 | Download Qwen2.5-0.5B-Instruct | short / c0611 | completed | 22s | 954 MB |
| 11310786 | Download Qwen2.5-Coder-3B-Instruct | short / c0633 | completed | 53s | 5.8 GB |
| 11310787 | Download Qwen2.5-Coder-7B-Instruct | short / c0639 | completed | 53s | 15 GB |
| 11310887 | Download + unzip datasets (first try) | short / c0611 | cancelled | 5m17s | Would have unpacked a 23 GB file and ~290k index files we do not use. F#3 |
| 11311119 | Download + unzip datasets (SkyRL-SQL-653, SkyRL-SQL-eval, GSM8K, OmniSQL databases) | short / c0611 | completed | 17m23s | 44 GB of databases (BIRD 32 GB, SynSQL 9.9 GB, Spider 1.8 GB) |
| 11311278 | Smoke: GSM8K GRPO, Qwen2.5-0.5B, 20 steps | gpu-short / d4054 | failed | 2m07s | verl workers refused ROCR_VISIBLE_DEVICES set by Slurm. F#10 |
| 11311626 | Same smoke, copy on gpu partition (A100) | gpu | cancelled | — | Not needed after 11313187 passed |
| 11311687 | Download Qwen2.5-Coder-32B-Instruct (SFT teacher) | short / c0658 | completed | 3m41s | — |
| 11311750 | Template check: LLaMA-Factory qwen vs HF chat template on a tool-calling conversation | short / c0591 | completed | 1m48s | Identical, 418/418 tokens |
| 11311942 | Build processed data (first version) | short / c0591 | completed | 4m14s | All 6 eval sets aligned to the originals with no gold-SQL mismatch; no missing DBs; no train/eval overlap; 1 BIRD dev gold fails → 1533. F#5, F#7 |
| 11311947 | Render training question 0 as the model sees it; run its gold SQL | short / c0591 | completed | 24s | 2456 prompt tokens; gold result **0 rows** (date drift). Train prompt tokens: median 1581, p90 2502, max 4460. F#6 |
| 11312004 | Baseline eval, Qwen2.5-Coder-3B-Instruct (A100) | gpu / d1028 | failed | 13m34s | Queued before the vLLM flag fix, ran the stale script. F#8, F#14 |
| 11312005 | Baseline eval, Qwen2.5-Coder-7B-Instruct | gpu / d4054 | failed | 16m32s | `--disable-log-requests` removed in vLLM 0.26. F#8 |
| 11312021 | Gold audit: empty and date-dependent gold results, all splits | short / c0591 | completed | 1m14s | Train 279/621 empty (45%), 22 date-dependent; Spider dev 49/1034; Spider test 54/2147; Spider-DK 50/535; Spider-Syn 46/1034; Spider-Realistic 32/508; BIRD dev 0/1533 empty, 16 date-dependent. F#6 |
| 11312082 | Rebuild processed data: train on non-empty golds, keep `*_all` for the ablation | short / c0591 | completed | 3m08s | train 342, val 21, train_all 621, val_all 32, val_monitor 521; eval sets unchanged |
| 11313165 | Baseline eval, 7B (second try) | gpu / d4054 | completed, **invalid** | 9m10s | All 6791 requests went through the site HTTP proxy and failed; the summary of zeros was deleted. F#12 |
| 11313170 | Duplicate of 11313165 | gpu | cancelled | — | My duplicate submission. F#11 |
| 11313187 | Smoke: GSM8K GRPO, Qwen2.5-0.5B, 20 steps, sdpa, no padding removal | gpu-short / d4053 | completed | 24m19s | Val acc 0 → 0.242 (step 10) → **0.453** (step 20); train score 0.019 → 0.444; ~40 s/step (gen 4 s, update 19 s, weight sync 0.47 s); peak 30 GB; vLLM vs FSDP prob. corr. 0.9995 |
| 11313191 | Duplicate of 11313187 | gpu-short | cancelled | — | My duplicate submission. F#11 |
| 11313297 | Baseline eval, 7B (third try, proxy bypassed) | gpu / d4054 | completed | 13m03s | Kept as `results/qwen7b-base-v1`. EX: Spider dev 0.297, test 0.290, DK 0.265, Syn 0.270, Realistic 0.321, BIRD dev 0.091; answered 22-40%; ~0 tool calls; 0 request errors; 161 s for 6791 questions |
| 11313450 | SQL smoke: multi-turn GRPO, 3B, 16 questions/step, 3 steps | gpu-short / d4054 | failed | 12m14s | Hydra: searchpath in base.yaml. F#13 |
| 11313640 | Baseline eval, 7B, with EX_lenient and saved non-answers | gpu / d4053 | completed | 9m45s | `results/qwen7b-base`. EX / lenient: Spider dev 0.309 / 0.510, test 0.287 / 0.494, DK 0.292 / 0.452, Syn 0.265 / 0.457, Realistic 0.295 / 0.472, BIRD dev 0.098 / 0.319. 4407 format failures, 2755 with a ```sql block; others use `<tool>` instead of `<tool_call>` or `<sql>` instead of `<solution>`. Greedy run-to-run difference vs 11313297: up to ~1.2 points |
| 11313809 | SQL smoke, second try | gpu-short / d4054 | failed | 7m46s | Step-0 validation ran with our tool and reward (3B base: format_ok 2.4% Spider dev, 18% BIRD dev, 0% train slice; all trajectories 2 turns, no parsed tool call); then OOM in compute_log_prob. F#15 |
| 11313950 | Baseline eval, 3B (resubmitted with the fixed script) | gpu (A100) / d1028 | completed | 211 s of eval | `results/qwen3b-base`. EX / lenient: Spider dev 0.004 / 0.057, test 0.006 / 0.062, DK 0.002 / 0.035, Syn 0.008 / 0.118, Realistic 0.006 / 0.053, BIRD dev 0.003 / 0.083. Answered (valid `<solution>`) only 0.4-2.5%; 0 tool calls; 6695 of 6791 are format failures (1610 with a ```sql block). 0 request errors |
| 11314015 | SFT teacher sampling: 32B on train.parquet, n=4, T=0.7, full conversations | gpu / d4053 | completed | 211 s of sampling | 1368 trajectories, 0 request errors. **0 parsed tool calls**: the 32B writes bare JSON `{"name": "execute_sql", ...}` without `<tool_call>` tags, so vLLM's hermes parser does not see a call. Stop reasons: format 742, answered 617, length 9. EX 0.222 (n=4 first sample); 177/342 questions solved in at least one sample (strict), all without tool use. Unusable for SFT as is (filter needs >=1 tool call) |
| 11314020 | SQL smoke, third try: fixed micro-batches of 2, chunked entropy | gpu-short / d4054 | completed | 8m28s | 3 steps of 16 questions x 5 samples, checkpoint `global_step_3` saved. Step 1: 80 trajectories scored -1 x68, 0 x9, 1 x3; all 2 turns (no `<tool_call>` ever; the model echoes the end of the tool definition `</tools>` and invents a `<query>` tag); 12/80 contain `<solution>`. Train score -0.8125 (step 1) -> -0.675 (step 3). Val acc on the 64-question sample: Spider dev 0.024 -> **0.233**, BIRD dev 0.091 -> 0.053, train slice 0 -> 0. Step 1 timing: 48 s (gen 4 s, old_log_prob 12 s, update 31 s, weight sync 0.6 s). A `DataLoader worker killed` traceback at interpreter exit, after training and W&B had finished, is shutdown noise |
| 11314299 | **Main run** `qwen3b_mt` leg 1: multi-turn GRPO, Qwen2.5-Coder-3B-Instruct, 342 non-empty-gold questions, 128/step x 5 samples, ~80 steps planned, sdpa (no flash-attn yet) | gpu (H200) / d4054 | running since 2026-10-22 22:41:44 | — | — |
| 11314384 | `qwen3b_mt` leg 2 (afterany 11314299; resumes from the newest checkpoint) | gpu (H200) | pending | — | Created although sbatch reported an error. F#16 |
| 11314602 | SFT teacher sampling, second try: 32B on train.parquet, n=4, T=0.7, `--lenient-tool-calls` (unwrapped calls are executed and stored as proper tool calls; first try kept as `results/teacher32b-train-v1`) | gpu (H200) | completed | — | `results/teacher32b-train`. 0 request errors. EX (first sample) **0.558** vs 0.222 without tool use. Stop: answered 1276, format 49, length 38, max_tool_calls 5. Tool calls per trajectory: 0 x625, 1 x340, 2 x183, 3 x153, 4 x50, 5 x17. 707 strict-correct trajectories, 408 of them with tool use; 191/342 questions solved with tool use (238 solved at all). Some call turns contain a literal `<|im_start|>` before the JSON (dropped by the sharegpt export, which keeps only the call arguments). `loose_tool_calls` over-counts calls on the last allowed turn (bookkeeping only) |

**Note on the main run (2026-10-22 22:45).** Qwen2.5-Coder 3B, 7B and 32B all fail to emit the `<tool_call>` wrapper the chat template asks for (3B: invents `<query>`; 7B: `<tool>`; 32B: bare JSON). verl's rollout uses the same strict hermes parser, so `qwen3b_mt` (RL from the instruct model) may never get a tool result back and only improve single-turn answers. Watch `num_turns/mean` (2 = never called the tool). The SFT-first run (teacher trajectories in the proper format) is the comparison that tests whether multi-turn tool use can be learned.
| 11314299 (step 0) | Main run step-0 validation (greedy, before any update) | — | — | — | Spider dev 0.073 (300-question sample), BIRD dev 0.055 (200), held-out train 0.095 (21); Spider dev valid format 9.3%; all trajectories 2 turns. These disagree with the eval harness (3B: Spider dev 0.004, valid format 0.8%), which led to F#17 |
| 11315211 | Baseline eval 3B, rerun with `--generation-config vllm` (no repetition penalty). Previous run kept as `results/qwen3b-base-genconfig` | gpu (A100) | pending | — | — |
| 11315228 | Baseline eval 7B, rerun with `--generation-config vllm`. Previous run kept as `results/qwen7b-base-genconfig` | gpu (H200) | pending | — | — |
| 11315262 | Install the flash-attn wheel into `envs/fa-r1` (used only by jobs with `SAR_USE_FA=1`, so the sdpa main run keeps the same kernels across legs) | short | submitted | — | — |
| login node | Build SFT data (`sar.data.sft`, per question the 2 shortest strict-correct tool-using trajectories, no leaked template tokens in kept text) from teacher run 11314602 | — | done | — | `data/processed/sft/sql_agent_sft.json`: **286 examples from 178 questions**; tool calls before the answer: 1 x135, 2 x70, 3 x66, 4 x10, 5 x5 |
