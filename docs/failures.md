# Failures and fixes

Every failure gets an entry, including my own command mistakes. Status is
`open` until the fix is verified.

| # | Date | What failed | Cause | Fix | Status |
|---|------|-------------|-------|-----|--------|
| 1 | 2026-10-22 | No flash-attn wheel for torch 2.11 | Upstream wheels stop at torch 2.8; verl's lock uses cu130 builds the cluster driver (570, CUDA 12.8) cannot run | Build from source on a CPU node for sm_80/sm_90 (`cluster/build_flash_attn.sbatch`); until then train with `attn_implementation=sdpa`, `use_remove_padding=False` | open |
| 2 | 2026-10-22 | verl 0.9.1's own pins (vLLM 0.24, cu130) do not run on driver 570 | CUDA 13 wheels | vLLM 0.26.0 cu129 + torch 2.11.0 cu129 (runs on 570 via CUDA minor-version compatibility, as verified in vllm-sglang-infra), verl installed without backend extras | verified: all packages import together (setup_env 11310779) |
| 3 | 2026-10-22 | First data job (11310887) would have unpacked all of OmniSQL's data.zip to node /tmp | I did not check the zip contents first: it holds a 23 GB `train_synsql.json` and ~290k index files we do not use | Cancelled; `download_data.sbatch` now excludes them | verified (11311119 running) |
| 4 | 2026-10-22 | A `sed` edit of `cluster/common.sh` failed (`bad flag in substitute command`) | My command: `#` delimiter clashed with a `#` in the replacement text | Edited the file directly | verified |
| 5 | 2026-10-22 | SkyRL-SQL-eval and OmniSQL eval json files have no `db_id` | Format only keeps prompt fields | Align by index with the original benchmark files and require the gold SQL to match on every row (`sar.data.build.load_eval`) | verified: all 6 eval sets aligned with no mismatch (build 11311942) |
| 6 | 2026-10-22 | Training question 0's gold SQL returns 0 rows today | The gold uses `strftime('%Y','now')`; the data was made in 2025 and the database holds metrics up to ~2022, so "last year" (2025) matches nothing. Any query returning no rows would score 1 | Audit (`tools/audit_gold.py`, job 11312021): **279 of 621 training golds (45%) return no rows**, only 18 of them through date drift; the rest are SynSQL's sparse synthetic databases. Eval sets: 3-9% empty (Spider family), 0 in BIRD dev (16 BIRD golds are date-dependent). Main runs train on non-empty golds (`train.parquet`); `train_all.parquet` keeps all 621 for the ablation `qwen3b_mt_alldata`; reward and eval report `pred_empty` / `gold_empty`, eval adds `EX_nonempty_gold` | fix in place, ablation pending |
| 7 | 2026-10-22 | 1 of 1534 BIRD dev gold queries fails to run | Data issue upstream | Dropped by `sar.data.build` validation; BIRD dev is reported on 1533 questions | verified (build 11311942) |
