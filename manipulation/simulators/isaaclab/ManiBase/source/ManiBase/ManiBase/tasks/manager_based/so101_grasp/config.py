"""SO101 抓取任务 YAML 配置读取。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ISAACLAB_DIR = Path(__file__).resolve().parents[7]
_CONFIG_PATH = _ISAACLAB_DIR / "config.yaml"

_DEFAULT_CFG: dict[str, Any] = {
    "scene": {"num_envs": 1024, "env_spacing": 1.2},
    "robot": {"mjcf_path": "../../../robots/assets/So101/mjcf/so101_new_calib.xml"},
    "cube": {"size": 0.035, "mass": 0.05, "initial_pos": [0.18, 0.0, 0.2975]},
    "table": {"size": [0.45, 0.60, 0.04], "pos": [0.16, 0.0, 0.255]},
    "episode": {"length_s": 6.0, "decimation": 2, "sim_dt": 0.01},
    "reward": {
        "reach_weight": -2.0,
        "lift_weight": 8.0,
        "grasp_weight": 2.0,
        "success_bonus": 5.0,
        "action_rate_weight": -0.01,
        "contact_force_threshold": 0.3,
    },
    "success": {"height_threshold": 0.08},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_so101_grasp_cfg() -> dict[str, Any]:
    cfg = deepcopy(_DEFAULT_CFG)
    if not _CONFIG_PATH.exists():
        return cfg
    data = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    override = data.get("so101_grasp", {})
    if override:
        cfg = _deep_update(cfg, override)
    return cfg


SO101_GRASP_CFG = load_so101_grasp_cfg()
