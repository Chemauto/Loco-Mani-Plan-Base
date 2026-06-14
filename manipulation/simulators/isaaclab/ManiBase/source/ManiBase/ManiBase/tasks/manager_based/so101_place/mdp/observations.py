"""SO101 抓取→放置任务：观测函数。

包含 / Includes:
    - ee_to_object_vector / ee_position_world：沿用自 so101_grasp（末端↔cube 误差、末端世界坐标）
    - cube_pose_in_robot_frame：cube 在机器人根坐标系的位姿（特权，供 critic）
    - command_target：从 CubePlaceCommand 读取放置目标点
    - ResNet10Extractor：相机图像特征提取（供 student 视觉变体）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_to_object_vector(env, ee_frame_cfg, object_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器指向物体的向量（世界系）/ Vector from end-effector to object (world frame).

    作用 / Purpose:
        提供 reward 优化的误差方向，是策略最重要的引导信号。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :]
    return obj.data.root_pos_w[:, :3] - ee_pos


def ee_position_world(env, ee_frame_cfg, ee_frame_index: int = 0) -> torch.Tensor:
    """末端执行器的世界坐标 / End-effector position in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, ee_frame_index, :]


def cube_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """cube 在机器人根坐标系下的位姿（位置 + 四元数）/ Cube pose in robot root frame (pos + quat).

    作用 / Purpose:
        特权观测，供 critic 使用（部署时 student 看不到，靠图像推断）。
    逻辑 / Logic:
        用 subtract_frame_transforms 把 cube 世界位姿转到机器人根坐标系。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos_b, cube_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        cube.data.root_pos_w[:, :3],
        cube.data.root_state_w[:, 3:7],
    )
    return torch.cat([cube_pos_b, cube_quat_b], dim=-1)


def command_target(env: ManagerBasedRLEnv, command_name: str = "object_place") -> torch.Tensor:
    """放置目标点（环境坐标系）/ Placement target point in env frame.

    作用 / Purpose:
        从命令系统读取目标点，让 critic/policy 知道 cube 要送到哪里。
    """
    return env.command_manager.get_command(command_name)


class ResNet10Extractor(ManagerTermBase):
    """相机图像 ResNet10 特征提取器（供 student 视觉变体）/ ResNet10 feature extractor for student vision variant.

    注意 / Note:
        teacher 状态版不使用此类；student 变体实例化时才加载模型。
        实现为 ManagerTermBase 风格（__init__ + __call__），student 变体接入时启用。
    """

    def __init__(self, cfg, env):
        from transformers import AutoModel

        super().__init__(cfg, env)
        model_name = getattr(cfg, "model_name", "helper2424/resnet10")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(env.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=env.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=env.device)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
        from isaaclab.envs.mdp.observations import image

        image_data = image(env=env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False)
        image_data = image_data.permute(0, 3, 1, 2).float() / 255.0
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        image_data = (image_data - mean) / std
        with torch.no_grad():
            # resnet10 输出 last_hidden_state (B, 512, 7, 7)（无 pooler_output），全局平均池化到 (B, 512)
            out = self.model(image_data)
            features = out.last_hidden_state.mean(dim=[2, 3])
        return features
