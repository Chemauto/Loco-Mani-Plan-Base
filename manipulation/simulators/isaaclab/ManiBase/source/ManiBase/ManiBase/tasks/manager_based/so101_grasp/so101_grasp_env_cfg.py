"""SO101 方块抓取强化学习任务。"""

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
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
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


@configclass
class So101GraspSceneCfg(InteractiveSceneCfg):
    """SO101 抓取场景。"""

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

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # 只上报活动夹爪指 <-> Cube 的接触（filter 排除桌面等误报）
    jaw_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SO101_JAW_LINK_NAME,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=0,
        track_air_time=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=1200.0),
    )


@configclass
class ActionsCfg:
    """动作空间。"""

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


@configclass
class ObservationsCfg:
    """观测空间。"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 夹爪开合状态（动作里有二值夹爪，策略必须能观测到它）
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SO101_GRIPPER_JOINT_NAME])},
        )
        # 末端 -> cube 的误差向量（reward 优化的方向），以及末端世界坐标
        ee_to_cube = ObsTerm(
            func=mdp.ee_to_object_vector,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
        )
        ee_pos = ObsTerm(func=mdp.ee_position_world, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """重置事件。"""

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


@configclass
class RewardsCfg:
    """奖励项。"""

    reach_cube = RewTerm(
        func=mdp.object_ee_distance,
        weight=_REWARD_CFG["reach_weight"],
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
    )
    # 夹爪指接触 cube 的密集奖励：reach -> lift 之间缺失的"碰到/夹住"信号
    grab_cube = RewTerm(
        func=mdp.gripper_object_contact,
        weight=_REWARD_CFG["grasp_weight"],
        params={
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names=SO101_JAW_LINK_NAME),
            "threshold": _REWARD_CFG["contact_force_threshold"],
        },
    )
    # 抬升高度仅在夹住时计奖（修复未抓取也白拿 lift 奖励的泄漏）
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
    lift_success = RewTerm(
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


@configclass
class TerminationsCfg:
    """终止条件。"""

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
        self.decimation = _EPISODE_CFG["decimation"]
        self.episode_length_s = _EPISODE_CFG["length_s"]
        self.viewer.eye = (0.65, -0.9, 0.65)
        self.viewer.lookat = (0.12, 0.0, 0.25)
        self.sim.dt = _EPISODE_CFG["sim_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class So101GraspCubeEnvCfg_PLAY(So101GraspCubeEnvCfg):
    """单环境调试配置。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.2
