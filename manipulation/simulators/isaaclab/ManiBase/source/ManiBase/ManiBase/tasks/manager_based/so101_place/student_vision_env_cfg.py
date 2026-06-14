"""SO101 抓取→放置：Student 视觉变体（teacher→student 蒸馏用）。

逻辑 / Logic:
    student policy 只看本体感觉 + 双相机 ResNet 特征（**无 ee_to_cube**），
    靠图像推断 cube 位置——这是 sim2-real 部署用的策略。
    teacher 组供蒸馏时 teacher 推理用（状态 + ee_to_cube 特权）。
    critic 组训练时辅助 value（含 ground-truth 特权）。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ManiBase.robots import (
    SO101_BASE_LINK_NAME,
    SO101_EE_LINK_NAME,
    SO101_FOLLOWER_CFG,
    SO101_GRIPPER_JOINT_NAME,
)

from . import mdp
from .config import SO101_PLACE_CFG
from .so101_place_env_cfg import So101PlaceEnvCfg

_CUBE_CFG = SO101_PLACE_CFG["cube"]


@configclass
class StudentObservationsCfg:
    """student 视觉变体观测（3 组：student policy / teacher / critic）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """student actor（部署用，无特权）：本体感觉 + 双相机 ResNet 特征。

        注意 / Note: 不含 ee_to_cube，cube 位置只能从图像推断。
        """
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SO101_GRIPPER_JOINT_NAME])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        last_action = ObsTerm(func=mdp.last_action)
        jaw_img = ObsTerm(
            func=mdp.ResNet10Extractor,
            params={"sensor_cfg": SceneEntityCfg("jaw_camera"), "data_type": "rgb"},
        )
        scene_img = ObsTerm(
            func=mdp.ResNet10Extractor,
            params={"sensor_cfg": SceneEntityCfg("scene_camera"), "data_type": "rgb"},
        )

        def __post_init__(self) -> None:
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """teacher 推理用（蒸馏时 teacher 在 env 内推理）：状态 + ee_to_cube 特权。"""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SO101_GRIPPER_JOINT_NAME])},
        )
        ee_to_cube = ObsTerm(
            func=mdp.ee_to_object_vector,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic（训练辅助）：ground-truth 特权，不噪声。"""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SO101_GRIPPER_JOINT_NAME])},
        )
        ee_to_cube = ObsTerm(
            func=mdp.ee_to_object_vector,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
        )
        last_action = ObsTerm(func=mdp.last_action)
        cube_pose = ObsTerm(func=mdp.cube_pose_in_robot_root_frame)
        place_target = ObsTerm(func=mdp.command_target, params={"command_name": "object_place"})
        ee_pos = ObsTerm(func=mdp.ee_position_world, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})

        def __post_init__(self) -> None:
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class So101PlaceStudentVisionEnvCfg(So101PlaceEnvCfg):
    """Student 视觉变体：双相机 + student 观测组，供 teacher→student 蒸馏。"""

    def __post_init__(self) -> None:
        super().__post_init__()  # 基础物理参数；相机保持启用（基础 SceneCfg 默认）

        # 机器人 / Robot
        self.scene.robot = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 目标物体 / Cube（同 teacher 物理参数）
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

        # 末端坐标系 / End-effector frame
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

        # 替换观测组为 student 版（双相机 ResNet + teacher/critic）/ replace with student obs groups
        self.observations = StudentObservationsCfg()


@configclass
class So101PlaceStudentVisionEnvCfg_PLAY(So101PlaceStudentVisionEnvCfg):
    """单环境回放配置 / Single-env playback config."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.2
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_ee_mass = None
