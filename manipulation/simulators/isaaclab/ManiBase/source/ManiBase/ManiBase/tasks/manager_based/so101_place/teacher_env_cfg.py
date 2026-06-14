"""SO101 抓取→放置任务：Teacher 变体（纯状态策略）。

逻辑 / Logic:
    填充基础 EnvCfg 的 MISSING：SO101 机器人 + cube + ee_frame。
    观测沿用基础 EnvCfg 的 teacher 组（policy 含特权 ee_to_cube）。
    这是蒸馏的"老师"，RL 直接训练即可收敛。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ManiBase.robots import SO101_BASE_LINK_NAME, SO101_EE_LINK_NAME, SO101_FOLLOWER_CFG

from .config import SO101_PLACE_CFG
from .so101_place_env_cfg import So101PlaceEnvCfg

_CUBE_CFG = SO101_PLACE_CFG["cube"]


@configclass
class So101PlaceTeacherEnvCfg(So101PlaceEnvCfg):
    """Teacher 变体：填充 robot/cube/ee_frame。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # teacher 是纯状态策略，禁用相机（省显存/加速）/ disable cameras for state-only teacher
        self.scene.jaw_camera = None
        self.scene.scene_camera = None

        # 机器人 / Robot
        self.scene.robot = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 目标物体 / Cube（沿用 so101_grasp 物理参数）
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(_CUBE_CFG["initial_pos"]), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=tuple([_CUBE_CFG["size"]] * 3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=_CUBE_CFG["mass"]),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 0.05)),
            ),
        )

        # 末端坐标系 / End-effector frame（base_link -> gripper_frame_link）
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_BASE_LINK_NAME}",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_EE_LINK_NAME}",
                    name="gripper",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )


@configclass
class So101PlaceTeacherEnvCfg_PLAY(So101PlaceTeacherEnvCfg):
    """单环境回放配置 / Single-env playback config."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.2
        # 关闭噪声与域随机化 / disable noise & domain randomization
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_ee_mass = None
