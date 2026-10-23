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
| 11310774 | Build flash-attn 2.8.3 wheel for torch 2.11 + CUDA 12 (sm_80, sm_90) | short / d0146 | running | 2h37m+ | — |
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
| 11313950 | Baseline eval, 3B (resubmitted with the fixed script) | gpu (A100) | running | — | — |
| 11314015 | SFT teacher sampling: 32B on train.parquet, n=4, T=0.7, full conversations | gpu | pending | — | — |
| 11314020 | SQL smoke, third try: fixed micro-batches of 2, chunked entropy | gpu-short / d4054 | completed | 8m28s | 3 steps of 16 questions x 5 samples, checkpoint `global_step_3` saved. Step 1: 80 trajectories scored -1 x68, 0 x9, 1 x3; all 2 turns (no `<tool_call>` ever; the model echoes the end of the tool definition `</tools>` and invents a `<query>` tag); 12/80 contain `<solution>`. Train score -0.8125 (step 1) -> -0.675 (step 3). Val acc on the 64-question sample: Spider dev 0.024 -> **0.233**, BIRD dev 0.091 -> 0.053, train slice 0 -> 0. Step 1 timing: 48 s (gen 4 s, old_log_prob 12 s, update 31 s, weight sync 0.6 s). A `DataLoader worker killed` traceback at interpreter exit, after training and W&B had finished, is shutdown noise |
| 11314299 | **Main run** `qwen3b_mt` leg 1: multi-turn GRPO, Qwen2.5-Coder-3B-Instruct, 342 non-empty-gold questions, 128/step x 5 samples, ~80 steps planned, sdpa (no flash-attn yet) | gpu (H200) | pending | — | — |
| 11314384 | `qwen3b_mt` leg 2 (afterany 11314299; resumes from the newest checkpoint) | gpu (H200) | pending | — | Created although sbatch reported an error. F#16 |
