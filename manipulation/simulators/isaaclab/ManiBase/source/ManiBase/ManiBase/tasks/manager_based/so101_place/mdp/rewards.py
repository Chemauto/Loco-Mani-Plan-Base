"""SO101 抓取→放置任务：奖励函数。

设计 / Design:
    reach -> grab -> lift -> place 四段串行，exp 核为主，关键门控防泄漏：
    - lift 必须夹住才计奖（防"未抓白拿抬升奖励"）
    - place 必须夹住且抬起才计奖（防"未搬运白拿 place 奖励"）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#-------------------------------------------------------------------------------------
# 辅助 / Helpers
#-------------------------------------------------------------------------------------
def _object_lift_height(env, object_cfg, robot_cfg, robot_base_name: str) -> torch.Tensor:
    """物体相对机器人基座的抬升高度，低于基座时置零 / Lift height relative to robot base."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    obj_height = obj.data.root_pos_w[:, 2]
    base_height = robot.data.body_pos_w[:, base_index, 2]
    return torch.clamp(obj_height - base_height, min=0.0)


def _gripper_object_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """活动夹爪指与目标物体的接触（ContactSensor 已 filter 只报 jaw<->cube）/ jaw-cube contact.

    返回每环境 0/1：接触力范数超过 threshold 即记 1。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.linalg.vector_norm(force, dim=-1) > threshold
    while contact.dim() > 1:
        contact = torch.any(contact, dim=1)
    return contact.float()


#-------------------------------------------------------------------------------------
# 任务奖励（exp 核 + 门控）/ Task rewards
#-------------------------------------------------------------------------------------
def object_ee_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_index: int = 0,
) -> torch.Tensor:
    """末端到 cube 的距离奖励（exp 核）/ EE-to-cube distance reward (exp kernel).

    逻辑 / Logic: exp(-dist/std)，距离越小奖励越大，稠密且有界。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_pos, dim=1)
    return torch.exp(-dist / std)


def gripper_grab_object(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """夹爪指接触 cube 的密集奖励 / Dense reward for jaw-cube contact.

    作用 / Purpose: reach -> lift 之间缺失的"碰到/夹住"信号。
    """
    return _gripper_object_contact(env, sensor_cfg, threshold)


def object_lift_height_when_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    robot_base_name: str = "base_link",
    threshold: float = 1.0,
) -> torch.Tensor:
    """物体抬升高度，仅在夹住时计奖 / Lift height gated by grasp.

    修复泄漏 / Fix leak: cube 初始就高于基座，未抓取也会白拿奖励；这里乘以接触门控。
    """
    height = _object_lift_height(env, object_cfg, robot_cfg, robot_base_name)
    grasping = _gripper_object_contact(env, sensor_cfg, threshold)
    return height * grasping


def object_place_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("jaw_contact_forces"),
    robot_base_name: str = "base_link",
    height_threshold: float = 0.08,
    threshold: float = 1.0,
) -> torch.Tensor:
    """cube 到放置目标的距离奖励（exp 核），门控夹住 / cube-to-target distance gated by grasp.

    逻辑 / Logic:
        只保留 grasping 门控（去掉 lifted 门控）。原来的 lifted&grasping 双门控太严，
        place_track 几乎恒为 0，策略拿不到 place 信号、卡在 reach/grab/lift 局部最优。
        放宽到“夹住就把 cube 送到目标”，先让 place 学起来，后续再收紧。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    # 命令是环境系目标点，转世界系 / command is target in env frame -> world frame
    target_w = env.command_manager.get_command(command_name) + env.scene.env_origins
    dist = torch.norm(target_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    grasping = _gripper_object_contact(env, sensor_cfg, threshold)
    return torch.exp(-dist / std) * grasping.float()


def place_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    dist_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """cube 到达目标点的一次性奖励 / One-shot bonus when cube reaches target."""
    obj: RigidObject = env.scene[object_cfg.name]
    target_w = env.command_manager.get_command(command_name) + env.scene.env_origins
    dist = torch.norm(target_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    return (dist < dist_threshold).float()


#-------------------------------------------------------------------------------------
# 碰撞惩罚 / Collision penalties（teacher 版可选接入）
#-------------------------------------------------------------------------------------
def table_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """夹爪/臂与桌面接触惩罚（接触力范数超阈值）/ Penalty for gripper/arm-table contact."""
    return _gripper_object_contact(env, sensor_cfg, threshold)


def self_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """机器人自碰撞惩罚 / Penalty for robot self-collision."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    while is_contact.dim() > 1:
        is_contact = torch.any(is_contact, dim=1)
    return is_contact.float()
