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
    # 相机 / Camera
    "camera": {
        "jaw": {
            "pos": (0.0, 0.08, 0.18),
            "rot": (0.185, 0.020, 0.274, 0.943),
            "focal_length": 24.0,
            "resolution": 224,
        },
        "scene": {
            "pos": (0.55, -0.30, 0.50),
            "rot": (0.622, 0.615, 0.343, 0.341),
            "focal_length": 24.0,
            "resolution": 224,
        },
    },
    "resnet": {"model_name": "helper2424/resnet10"},
    # 课程学习 / Curriculum：ee 初始位置渐远（关节噪声从 0 → 0.3 rad）
    "curriculum": {
        "init_joint_noise": 0.0,
        "final_joint_noise": 0.3,
        "num_steps": 50000,
    },
    # 预抓取关节角 / Pre-grasp pose（ee 在 cube 正上方）
    "pre_grasp": {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.4,
        "elbow_flex": -0.6,
        "wrist_flex": -0.3,
        "wrist_roll": 0.0,
        "gripper": 0.8,
    },
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
