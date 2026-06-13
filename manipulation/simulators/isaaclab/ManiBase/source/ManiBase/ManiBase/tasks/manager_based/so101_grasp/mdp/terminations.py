"""SO101 抓取任务终止函数。"""

from __future__ import annotations

import torch


def object_height_above_base(env, object_cfg, robot_cfg, robot_base_name: str = "base", height_threshold: float = 0.08) -> torch.Tensor:
    """物体相对机器人基座抬升超过阈值时判定成功。"""
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    obj_height = obj.data.root_pos_w[:, 2]
    base_height = robot.data.body_pos_w[:, base_index, 2]
    return obj_height - base_height > height_threshold


def object_dropped(env, object_cfg, minimum_height: float = 0.02) -> torch.Tensor:
    """物体低于最低高度时终止。"""
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height
