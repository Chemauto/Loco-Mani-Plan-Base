"""SO101 抓取→放置任务：放置目标命令。

逻辑 / Logic:
    在桌面范围内随机采样 cube 的放置目标点，作为命令下发给策略。
    方块各向同性，不需要 yaw，命令仅为 (x, y, z) 目标位置。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CubePlaceCommand(CommandTerm):
    """在桌面范围内采样 cube 的放置目标位姿 / Sample target pose for placing the cube."""

    cfg: "CubePlaceCommandCfg"

    def __init__(self, cfg: "CubePlaceCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.cube: RigidObject = env.scene[cfg.asset_name]
        # 目标点：环境坐标系 / world 系
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_w = torch.zeros_like(self.pos_command_e)
        # 诊断指标：cube 当前位置到目标点的距离 / diagnostic: cube-to-target distance
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.initial_error_pos = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # 返回环境坐标系下的目标点 (E, 3) / target point in env frame
        return self.pos_command_e

    def _update_metrics(self):
        # cube 当前位置到目标点（世界系）的 xy 距离 / xy distance cube->target in world
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w[:, :2] - self.cube.data.root_pos_w[:, :2], dim=1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_e[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_e[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_e[env_ids, 2] = self.cfg.place_height
        # 环境系 → 世界系（加环境原点偏移）/ env frame -> world frame
        self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
        # 记录初始误差，供课程计算进度 / record initial error for curriculum progress
        self.initial_error_pos[env_ids] = torch.norm(
            self.pos_command_w[env_ids, :2] - self.cube.data.root_pos_w[env_ids, :2], dim=1
        )

    def _update_command(self):
        # 静态目标，episode 内不更新 / static target, no intra-episode update
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # 用一个小球标记目标点 / mark target point with a small frame
        self.goal_pose_visualizer.visualize(translations=self.pos_command_w)


@configclass
class CubePlaceCommandCfg(CommandTermCfg):
    """放置目标命令配置 / Configuration for the cube place target command."""

    class_type: type = CubePlaceCommand

    asset_name: str = MISSING

    # 放置高度（桌面 + cube 半高）/ placement height (table top + half cube)
    place_height: float = MISSING

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    ranges: "CubePlaceCommandCfg.Ranges" = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/cube_place_target"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
