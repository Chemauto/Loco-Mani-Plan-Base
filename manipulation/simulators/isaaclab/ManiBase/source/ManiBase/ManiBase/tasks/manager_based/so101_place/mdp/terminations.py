"""SO101 抓取→放置任务：终止条件。"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def object_at_target(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base_link",
    dist_threshold: float = 0.03,
    height_threshold: float = 0.08,
) -> torch.Tensor:
    """cube 到达放置目标且抬升达标 → 成功 / Success: cube at target and lifted.

    逻辑 / Logic:
        1. cube 到目标点 xy 距离 < dist_threshold
        2. cube 相对机器人基座抬升 > height_threshold（确认是搬运到位，而非滚到目标）
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    # 目标点世界系 / target in world frame
    target_w = env.command_manager.get_command(command_name) + env.scene.env_origins
    dist = torch.norm(target_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    # 抬升高度（相对基座）/ lift height relative to base
    base_index = robot.data.body_names.index(robot_base_name)
    height = obj.data.root_pos_w[:, 2] - robot.data.body_pos_w[:, base_index, 2]
    return (dist < dist_threshold) & (height > height_threshold)


def object_dropped(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    minimum_height: float = 0.02,
) -> torch.Tensor:
    """物体低于最低高度（掉桌）→ 失败 / Failure: object below minimum height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height
