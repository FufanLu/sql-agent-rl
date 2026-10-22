# Failures and fixes

Every failure gets an entry, including my own command mistakes. Status is
`open` until the fix is verified.

| # | Date | What failed | Cause | Fix | Status |
|---|------|-------------|-------|-----|--------|
| 1 | 2026-10-22 | No flash-attn wheel for torch 2.11 | Upstream wheels stop at torch 2.8; verl's lock uses cu130 builds the cluster driver (570, CUDA 12.8) cannot run | Build from source on a CPU node for sm_80/sm_90 (`cluster/build_flash_attn.sbatch`); until then train with `attn_implementation=sdpa`, `use_remove_padding=False` | open |
| 2 | 2026-10-22 | verl 0.9.1's own pins (vLLM 0.24, cu130) do not run on driver 570 | CUDA 13 wheels | vLLM 0.26.0 cu129 + torch 2.11.0 cu129 (runs on 570 via CUDA minor-version compatibility, as verified in vllm-sglang-infra), verl installed without backend extras | verified: all packages import together (setup_env 11310779) |
| 3 | 2026-10-22 | First data job (11310887) would have unpacked all of OmniSQL's data.zip to node /tmp | I did not check the zip contents first: it holds a 23 GB `train_synsql.json` and ~290k index files we do not use | Cancelled; `download_data.sbatch` now excludes them | verified (11311119 running) |
| 4 | 2026-10-22 | A `sed` edit of `cluster/common.sh` failed (`bad flag in substitute command`) | My command: `#` delimiter clashed with a `#` in the replacement text | Edited the file directly | verified |
| 5 | 2026-10-22 | SkyRL-SQL-eval and OmniSQL eval json files have no `db_id` | Format only keeps prompt fields | Align by index with the original benchmark files and require the gold SQL to match on every row (`sar.data.build.load_eval`) | open |
