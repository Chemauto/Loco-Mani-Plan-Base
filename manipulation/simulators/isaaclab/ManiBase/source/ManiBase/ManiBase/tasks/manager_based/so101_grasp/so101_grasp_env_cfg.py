"""SO101 方块抓取强化学习任务（双相机视觉 + 课程学习）。"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ManiBase.robots import (
    SO101_ARM_JOINT_NAMES,
    SO101_BASE_LINK_NAME,
    SO101_EE_LINK_NAME,
    SO101_FOLLOWER_CFG,
    SO101_GRIPPER_JOINT_NAME,
    SO101_JAW_LINK_NAME,
)

from . import mdp
from .config import SO101_GRASP_CFG

_CUBE_CFG = SO101_GRASP_CFG["cube"]
_TABLE_CFG = SO101_GRASP_CFG["table"]
_SCENE_CFG = SO101_GRASP_CFG["scene"]
_EPISODE_CFG = SO101_GRASP_CFG["episode"]
_REWARD_CFG = SO101_GRASP_CFG["reward"]
_SUCCESS_CFG = SO101_GRASP_CFG["success"]
_CAMERA_CFG = SO101_GRASP_CFG["camera"]
_PRE_GRASP = SO101_GRASP_CFG["pre_grasp"]


# ============================= 场景 / Scene =============================

@configclass
class So101GraspSceneCfg(InteractiveSceneCfg):
    """SO101 抓取场景：机器人 + 方块 + 桌面 + 双相机。"""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0)),
        collision_group=-1,
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(_TABLE_CFG["pos"]), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=tuple(_TABLE_CFG["size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
        ),
    )

    cube = RigidObjectCfg(
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

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={k: _PRE_GRASP[k] for k in SO101_ARM_JOINT_NAMES + [SO101_GRIPPER_JOINT_NAME]},
            joint_vel={".*": 0.0},
        ),
    )

    ee_frame = FrameTransformerCfg(
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

    jaw_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SO101_JAW_LINK_NAME,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=0,
        track_air_time=False,
    )

    # 相机暂不加入 scene（TODO: 加相机视觉时创建 _Vision 变体）

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=1200.0),
    )


# ============================= 动作 / Actions =============================

@configclass
class ActionsCfg:
    """动作空间：arm 关节位置 + gripper 二值开合。"""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_ARM_JOINT_NAMES,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[SO101_GRIPPER_JOINT_NAME],
        open_command_expr={SO101_GRIPPER_JOINT_NAME: 0.8},
        close_command_expr={SO101_GRIPPER_JOINT_NAME: -0.05},
    )


# ============================= 观测 / Observations =============================

@configclass
class ObservationsCfg:
    """非对称观测：policy（本体 + ee_to_cube + 双相机特征），critic（更多特权）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 观测（暂不含相机，先跑通 RL 再加）。"""
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
        # TODO: 加相机后取消注释
        # jaw_img = ObsTerm(
        #     func=mdp.ResNet10Extractor,
        #     params={"sensor_cfg": SceneEntityCfg("jaw_camera"), "data_type": "rgb"},
        # )
        # scene_img = ObsTerm(
        #     func=mdp.ResNet10Extractor,
        #     params={"sensor_cfg": SceneEntityCfg("scene_camera"), "data_type": "rgb"},
        # )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 观测（policy 全部 + 额外特权）。"""
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
        ee_pos = ObsTerm(func=mdp.ee_position_world, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        cube_pose = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("cube")},
        )
        # TODO: 加相机后取消注释
        # jaw_img = ObsTerm(
        #     func=mdp.ResNet10Extractor,
        #     params={"sensor_cfg": SceneEntityCfg("jaw_camera"), "data_type": "rgb"},
        # )
        # scene_img = ObsTerm(
        #     func=mdp.ResNet10Extractor,
        #     params={"sensor_cfg": SceneEntityCfg("scene_camera"), "data_type": "rgb"},
        # )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================= 事件 / Events =============================

@configclass
class EventCfg:
    """重置事件：cube 小范围随机化。"""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-0.4, 0.4)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


# ============================= 奖励 / Rewards =============================

@configclass
class RewardsCfg:
    """奖励：dense(reach+grab+lift) + sparse(success) + regularization。"""

    reach_cube = RewTerm(
        func=mdp.object_ee_distance,
        weight=_REWARD_CFG["reach_weight"],
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
    )
    grab_cube = RewTerm(
        func=mdp.gripper_object_contact,
        weight=_REWARD_CFG["grasp_weight"],
        params={
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names=SO101_JAW_LINK_NAME),
            "threshold": _REWARD_CFG["contact_force_threshold"],
        },
    )
    lift_cube = RewTerm(
        func=mdp.object_lift_height_when_grasped,
        weight=_REWARD_CFG["lift_weight"],
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names=SO101_JAW_LINK_NAME),
            "robot_base_name": SO101_BASE_LINK_NAME,
            "threshold": _REWARD_CFG["contact_force_threshold"],
        },
    )
    success = RewTerm(
        func=mdp.object_lift_success,
        weight=_REWARD_CFG["success_bonus"],
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "height_threshold": _SUCCESS_CFG["height_threshold"],
            "robot_base_name": SO101_BASE_LINK_NAME,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=_REWARD_CFG["action_rate_weight"])


# ============================= 终止 / Terminations =============================

@configclass
class TerminationsCfg:
    """终止条件：超时 / 成功 / cube 掉落。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.object_height_above_base,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "height_threshold": _SUCCESS_CFG["height_threshold"],
            "robot_base_name": SO101_BASE_LINK_NAME,
        },
    )
    cube_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"object_cfg": SceneEntityCfg("cube"), "minimum_height": 0.02},
    )


# ============================= 环境配置 =============================

@configclass
class So101GraspCubeEnvCfg(ManagerBasedRLEnvCfg):
    """SO101 方块抓取训练环境。"""

    scene: So101GraspSceneCfg = So101GraspSceneCfg(
        num_envs=_SCENE_CFG["num_envs"],
        env_spacing=_SCENE_CFG["env_spacing"],
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # 禁用 DLSS（分辨率 224 低于 DLSS 最小要求 300）
        import carb.settings
        _carb_settings = carb.settings.get_settings()
        _carb_settings.set("/rtx/post/dlss/enabled", False)

        self.decimation = _EPISODE_CFG["decimation"]
        self.episode_length_s = _EPISODE_CFG["length_s"]
        self.viewer.eye = (0.65, -0.9, 0.65)
        self.viewer.lookat = (0.12, 0.0, 0.25)
        self.sim.dt = _EPISODE_CFG["sim_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        # 8GB GPU：降低 physx buffer 以适应更多 envs
        self.sim.physx.gpu_max_rigid_contact_count = 2097152
        self.sim.physx.gpu_max_rigid_patch_count = 163840
        self.sim.physx.gpu_found_lost_pairs_capacity = 2097152


@configclass
class So101GraspCubeEnvCfg_PLAY(So101GraspCubeEnvCfg):
    """单环境回放配置。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.2
        self.observations.policy.enable_corruption = False
