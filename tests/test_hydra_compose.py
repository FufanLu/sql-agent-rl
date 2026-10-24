"""Composes every GRPO run config with verl's own config files, as
`python -m verl.trainer.main_ppo --config-name <run>` does on the cluster,
so a broken config fails here instead of 10 minutes into a GPU job.

Needs verl's config files (`pip install --no-deps verl==0.9.1`); importing
verl itself would pull in ray and torch, so its config directory is found
without importing it and referenced with file:// instead of pkg://.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.find_spec("verl")
VERL_CONFIG = Path(spec.submodule_search_locations[0]) / "trainer" / "config" if spec else None
hydra = pytest.importorskip("hydra")
pytestmark = pytest.mark.skipif(VERL_CONFIG is None or not VERL_CONFIG.is_dir(), reason="verl config files not installed")

RUNS = sorted(p.stem for p in (ROOT / "configs/grpo").glob("*.yaml") if p.stem != "base" and not p.stem.startswith("_"))


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    for k in ("SAR_DATA", "SAR_MODEL", "SAR_CODE", "SAR_CKPT_DIR"):
        monkeypatch.setenv(k, f"/x/{k.lower()}")
    out = tmp_path / "grpo"
    shutil.copytree(ROOT / "configs/grpo", out)
    for p in out.glob("*.yaml"):
        p.write_text(p.read_text().replace("pkg://verl.trainer.config", f"file://{VERL_CONFIG}"))
    return out


@pytest.mark.parametrize("run", RUNS)
def test_run_config_composes(config_dir, run):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=run)
    c = OmegaConf.to_container(cfg, resolve=True)
    assert c["algorithm"]["adv_estimator"] == "grpo"
    assert c["actor_rollout_ref"]["rollout"]["multi_turn"]["enable"] is True
    assert c["actor_rollout_ref"]["rollout"]["multi_turn"]["tool_config_path"] == "/x/sar_code/configs/tools/sql.yaml"
    assert c["reward"]["custom_reward_function"]["path"] == "/x/sar_code/src/sar/reward/sql_reward.py"
    assert c["trainer"]["n_gpus_per_node"] == 1
    assert c["data"]["train_files"].startswith("/x/sar_data/")
    model = c["actor_rollout_ref"]["model"]
    import yaml
    if "_sdpa" in yaml.safe_load((config_dir / f"{run}.yaml").read_text())["defaults"]:
        assert model["use_remove_padding"] is False and model["override_config"]["attn_implementation"] == "sdpa"
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
    fg = c["algorithm"]["filter_groups"]
    assert fg["enable"] is run.endswith("_dapo")
    if fg["enable"]:
        assert fg["metric"] == "reward"


def test_verl_sft_config_composes(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    for k in ("SAR_SFT_DATA", "SAR_MODEL", "SAR_CKPT_DIR"):
        monkeypatch.setenv(k, f"/x/{k.lower()}")
    out = tmp_path / "sft"
    shutil.copytree(ROOT / "configs/sft", out)
    p = out / "verl_sft.yaml"
    p.write_text(p.read_text().replace("pkg://verl.trainer.config", f"file://{VERL_CONFIG}"))
    with initialize_config_dir(config_dir=str(out), version_base=None):
        cfg = compose(config_name="verl_sft")
    c = OmegaConf.to_container(cfg, resolve=True)
    assert c["data"]["train_files"] == "/x/sar_sft_data/sft_messages.parquet"
    assert c["data"]["tools_key"] == "tools" and c["model"]["path"] == "/x/sar_model"
    assert c["model"]["use_remove_padding"] is True
    assert c["optim"]["lr"] == 1.0e-5 and c["trainer"]["total_epochs"] == 2


def test_flash_attn_switch_reads_the_defaults_list(tmp_path):
    """cluster/grpo.sbatch turns flash-attn on unless the run's defaults list
    includes _sdpa; a comment mentioning _sdpa must not count."""
    import re
    import subprocess

    line = next(x for x in (ROOT / "cluster/grpo.sbatch").read_text().splitlines() if "SAR_USE_FA=1" in x)
    pattern = re.search(r"grep -qE '([^']+)'", line).group(1)
    for run in RUNS:
        cfg = ROOT / "configs/grpo" / f"{run}.yaml"
        uses_sdpa = "_sdpa" in __import__("yaml").safe_load(cfg.read_text())["defaults"]
        found = subprocess.run(["grep", "-qE", pattern, str(cfg)]).returncode == 0
        assert found == uses_sdpa, run
