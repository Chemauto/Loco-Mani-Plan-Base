"""SO101 抓取任务观测函数。"""

from __future__ import annotations

import torch


def object_position_in_robot_root_frame(env, robot_cfg, object_cfg) -> torch.Tensor:
    """计算物体相对机器人根位置的向量。"""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3] - robot.data.body_pos_w[:, 0, :3]


def ee_to_object_vector(env, ee_frame_cfg, object_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器指向物体的向量（世界系），即 reward 优化的误差方向。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    return obj.data.root_pos_w[:, :3] - ee_pos


def ee_position_world(env, ee_frame_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器的世界坐标。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, ee_frame_index, :]
