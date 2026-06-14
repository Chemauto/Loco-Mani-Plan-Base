"""SO101 抓取→放置任务 YAML 配置读取。

逻辑 / Logic:
    1. 内嵌默认参数 _DEFAULT_CFG（开箱即用）/ Embed default params for out-of-the-box run.
    2. 若存在 config.yaml 的 so101_place 段，深合并覆盖 / Override with so101_place section in config.yaml.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ISAACLAB_DIR = Path(__file__).resolve().parents[7]
_CONFIG_PATH = _ISAACLAB_DIR / "config.yaml"

# 默认参数 / Default parameters
# 注意 / Note: camera / resnet 段供 student 视觉变体使用，teacher 状态版不读。
_DEFAULT_CFG: dict[str, Any] = {
    "scene": {"num_envs": 1024, "env_spacing": 1.2},
    "cube": {"size": 0.035, "mass": 0.05, "initial_pos": [0.18, 0.0, 0.2975]},
    "table": {"size": [0.45, 0.60, 0.04], "pos": [0.16, 0.0, 0.255]},
    "episode": {"length_s": 8.0, "decimation": 2, "sim_dt": 0.01},
    # 放置命令 / Place command（目标点在桌面范围内随机）
    # 范围收紧到 cube 初始位 (0.18, 0) 附近，让 place 奖励更容易触发，避免初始就太难
    "place": {
        "target_pos_x": (0.13, 0.23),   # 放置目标 x 范围（收紧，靠近 cube 初始 0.18）/ target x range
        "target_pos_y": (-0.05, 0.05),  # 放置目标 y 范围（收紧）/ target y range
        "place_height": 0.2975,          # 放置高度（桌面 + cube 半高）/ placement height
        "dist_threshold": 0.03,          # 到目标即成功的距离阈值 / success distance threshold
        "height_threshold": 0.08,        # 抬升成功阈值（相对基座）/ lift success height
    },
    # 课程 / Curriculum
    "curriculum": {
        "progress_beta": 0.02,           # EMA 平滑系数 / EMA smoothing
        "action_rate_num_steps": 14400,  # action_rate 惩罚收紧步数 / steps to tighten action_rate
    },
    # 域随机化 / Domain randomization
    "event": {
        "static_friction": (0.5, 1.2),
        "dynamic_friction": (0.5, 1.2),
        "ee_mass_range": (0.0, 0.5),
    },
    # 奖励 / Reward
    "reward": {
        "reach_weight": 2.0,
        "reach_std": 0.10,
        "grab_weight": 2.0,
        "grab_threshold": 0.3,
        "lift_weight": 8.0,
        "place_weight": 16.0,
        "place_std": 0.10,
        "success_bonus": 5.0,
        "action_rate_weight": -1.0e-4,
        "joint_vel_weight": -1.0e-4,
        "contact_force_threshold": 0.3,
    },
    # 相机 / Camera（student 视觉变体用）/ for student vision variant
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
    # 视觉特征提取器 / Vision feature extractor（student 视觉变体用）
    "resnet": {
        "model_name": "helper2424/resnet10",
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_so101_place_cfg() -> dict[str, Any]:
    cfg = deepcopy(_DEFAULT_CFG)
    if not _CONFIG_PATH.exists():
        return cfg
    data = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    override = data.get("so101_place", {})
    if override:
        cfg = _deep_update(cfg, override)
    return cfg


SO101_PLACE_CFG = load_so101_place_cfg()
