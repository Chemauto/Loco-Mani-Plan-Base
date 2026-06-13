"""SO101 抓取任务奖励函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor


def object_ee_distance(env, ee_frame_cfg, object_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """计算目标物体到末端执行器目标坐标系的距离。"""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    obj_pos = obj.data.root_pos_w[:, :3]
    return torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)


def object_lift_height(env, object_cfg, robot_cfg, robot_base_name: str = "base") -> torch.Tensor:
    """计算物体相对机器人基座的抬升高度，低于基座时置零。"""
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    obj_height = obj.data.root_pos_w[:, 2]
    base_height = robot.data.body_pos_w[:, base_index, 2]
    return torch.clamp(obj_height - base_height, min=0.0)


def object_lift_success(env, object_cfg, robot_cfg, robot_base_name: str = "base", height_threshold: float = 0.08) -> torch.Tensor:
    """物体超过成功高度时给额外奖励。"""
    height = object_lift_height(env, object_cfg, robot_cfg, robot_base_name)
    return (height > height_threshold).float()


def gripper_object_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """活动夹爪指与目标物体的接触（ContactSensor 已用 filter 只报 finger<->cube）。

    返回每环境 0/1：任一被追踪夹爪体对物体的接触力范数超过 threshold 即记 1。
    这是 reach->lift 之间缺失的密集"碰到/夹住"信号。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (E, B, 3) 或 (E, 3)
    contact = torch.linalg.vector_norm(force, dim=-1) > threshold  # (E, B) 或 (E,)
    # 把可能的 body 维度压掉，保证返回 (E,)
    while contact.dim() > 1:
        contact = torch.any(contact, dim=1)
    return contact.float()


def object_lift_height_when_grasped(
    env,
    object_cfg,
    robot_cfg,
    sensor_cfg: SceneEntityCfg,
    robot_base_name: str = "base_link",
    threshold: float = 1.0,
) -> torch.Tensor:
    """物体抬升高度，仅在夹住物体时计奖。

    修复原 object_lift_height 的泄漏：cube 初始就高于机器人基座，未抓取也会白拿奖励。
    这里乘以 gripper_object_contact 门控，只有真夹住才计抬升。
    """
    height = object_lift_height(env, object_cfg, robot_cfg, robot_base_name)
    grasping = gripper_object_contact(env, sensor_cfg, threshold)
    return height * grasping
