"""SO101 抓取→放置任务：基础环境配置。

任务流程 / Task flow: reach -> grab -> lift -> place（cube 送到放置目标点）。
本文件为基础 EnvCfg：robot/cube/ee_frame 用 MISSING 占位，由 teacher / student 变体填充。
观测组为 teacher 版（policy 含特权 ee_to_cube，critic 含更多 ground-truth）。
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ManiBase.robots import (
    SO101_ARM_JOINT_NAMES,
    SO101_BASE_LINK_NAME,
    SO101_EE_LINK_NAME,
    SO101_GRIPPER_JOINT_NAME,
    SO101_JAW_LINK_NAME,
)

from . import mdp
from .config import SO101_PLACE_CFG

# 参数分段 / Config sections
_SCENE_CFG = SO101_PLACE_CFG["scene"]
_CUBE_CFG = SO101_PLACE_CFG["cube"]
_TABLE_CFG = SO101_PLACE_CFG["table"]
_EPISODE_CFG = SO101_PLACE_CFG["episode"]
_PLACE_CFG = SO101_PLACE_CFG["place"]
_CURRICULUM_CFG = SO101_PLACE_CFG["curriculum"]
_EVENT_CFG = SO101_PLACE_CFG["event"]
_REWARD_CFG = SO101_PLACE_CFG["reward"]
_CAMERA_CFG = SO101_PLACE_CFG["camera"]


############################## 场景定义 / Scene ##############################
@configclass
class So101PlaceSceneCfg(InteractiveSceneCfg):
    """SO101 抓取→放置场景。"""

    # 地面 / Ground
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0)),
        collision_group=-1,
    )

    # 桌面 / Table（沿用 so101_grasp 配置）
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

    # 目标物体 / Target cube（占位，变体填充）
    cube: RigidObjectCfg = MISSING  # type: ignore[name-defined]

    # 机器人 / Robot（占位，变体填充）
    robot: ArticulationCfg = MISSING  # type: ignore[name-defined]

    # 末端坐标系 / End-effector frame（占位，变体填充）
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore[name-defined]

    # 夹爪指 <-> Cube 接触（filter 只报 jaw<->cube，用于 grab/lift/place 门控）
    jaw_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SO101_JAW_LINK_NAME,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=0,
        track_air_time=False,
    )

    # 手眼相机 / Eye-in-hand camera（TiledCamera 适配多环境，绕开 SyntheticData 限制；teacher 变体禁用）
    jaw_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SO101_JAW_LINK_NAME + "/JawCamera",
        data_types=["rgb"],
        height=_CAMERA_CFG["jaw"]["resolution"],
        width=_CAMERA_CFG["jaw"]["resolution"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_CAMERA_CFG["jaw"]["focal_length"],
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(_CAMERA_CFG["jaw"]["pos"]),
            rot=tuple(_CAMERA_CFG["jaw"]["rot"]),
            convention="opengl",
        ),
    )

    # 场景相机 / Scene camera（全局视角，看 cube 和目标点；teacher 变体禁用）
    scene_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/SceneCamera",
        data_types=["rgb"],
        height=_CAMERA_CFG["scene"]["resolution"],
        width=_CAMERA_CFG["scene"]["resolution"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_CAMERA_CFG["scene"]["focal_length"],
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(_CAMERA_CFG["scene"]["pos"]),
            rot=tuple(_CAMERA_CFG["scene"]["rot"]),
            convention="opengl",
        ),
    )

    # 光照 / Light
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=1200.0),
    )


############################## 动作 / Actions ##############################
@configclass
class ActionsCfg:
    """动作空间 / Action space（5 臂关节位置 + 二值夹爪）。"""

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


############################## 命令 / Commands ##############################
@configclass
class CommandsCfg:
    """放置目标命令 / Placement target command."""

    object_place = mdp.CubePlaceCommandCfg(
        asset_name="cube",
        place_height=_PLACE_CFG["place_height"],
        resampling_time_range=(_EPISODE_CFG["length_s"], _EPISODE_CFG["length_s"]),
        debug_vis=False,
        ranges=mdp.CubePlaceCommandCfg.Ranges(
            pos_x=_PLACE_CFG["target_pos_x"],
            pos_y=_PLACE_CFG["target_pos_y"],
        ),
    )


############################## 观测 / Observations（teacher 版） ##############################
@configclass
class ObservationsCfg:
    """观测空间 / Observation space（teacher 版：policy 含特权 ee_to_cube）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测（actor）：本体感觉 + 特权 ee_to_cube + 上次动作。"""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SO101_GRIPPER_JOINT_NAME])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        ee_to_cube = ObsTerm(
            func=mdp.ee_to_object_vector,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
        )  # 特权：末端->cube 误差
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic 观测（value）：policy 全部 + 更多 ground-truth，不噪声。"""
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
        cube_pose = ObsTerm(  # 特权：cube 完整位姿（机器人根坐标系）
            func=mdp.cube_pose_in_robot_root_frame,
        )
        place_target = ObsTerm(  # 特权：放置目标点
            func=mdp.command_target,
            params={"command_name": "object_place"},
        )
        ee_pos = ObsTerm(
            func=mdp.ee_position_world,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        def __post_init__(self) -> None:
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


############################## 奖励 / Rewards ##############################
@configclass
class RewardsCfg:
    """奖励项（grab->place 全流程，exp 核 + 门控）。"""

    ###############任务奖励（exp 核 + 门控）###############
    reach_cube = RewTerm(
        func=mdp.object_ee_distance_exp,
        weight=_REWARD_CFG["reach_weight"],
        params={"std": _REWARD_CFG["reach_std"], "ee_frame_cfg": SceneEntityCfg("ee_frame"), "object_cfg": SceneEntityCfg("cube")},
    )
    grab_cube = RewTerm(
        func=mdp.gripper_grab_object,
        weight=_REWARD_CFG["grab_weight"],
        params={"sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names=SO101_JAW_LINK_NAME), "threshold": _REWARD_CFG["grab_threshold"]},
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
    place_track = RewTerm(
        func=mdp.object_place_distance_exp,
        weight=_REWARD_CFG["place_weight"],
        params={
            "std": _REWARD_CFG["place_std"],
            "command_name": "object_place",
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("jaw_contact_forces", body_names=SO101_JAW_LINK_NAME),
            "robot_base_name": SO101_BASE_LINK_NAME,
            "height_threshold": _PLACE_CFG["height_threshold"],
            "threshold": _REWARD_CFG["contact_force_threshold"],
        },
    )
    place_bonus = RewTerm(
        func=mdp.place_success_bonus,
        weight=_REWARD_CFG["success_bonus"],
        params={"command_name": "object_place", "dist_threshold": _PLACE_CFG["dist_threshold"], "object_cfg": SceneEntityCfg("cube")},
    )

    ###############惩罚###############
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=_REWARD_CFG["action_rate_weight"])
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=_REWARD_CFG["joint_vel_weight"],
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


############################## 终止 / Terminations ##############################
@configclass
class TerminationsCfg:
    """终止条件 / Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    place_success = DoneTerm(
        func=mdp.object_at_target,
        params={
            "command_name": "object_place",
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": SO101_BASE_LINK_NAME,
            "dist_threshold": _PLACE_CFG["dist_threshold"],
            "height_threshold": _PLACE_CFG["height_threshold"],
        },
    )
    cube_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"object_cfg": SceneEntityCfg("cube"), "minimum_height": 0.02},
    )


############################## 课程 / Curriculum ##############################
@configclass
class CurriculumCfg:
    """课程项 / Curriculum terms（自适应扩大放置目标范围）。"""

    place_range = CurrTerm(
        func=mdp.place_target_progress_curriculum,
        params={"command_name": "object_place", "progress_beta": _CURRICULUM_CFG["progress_beta"]},
    )
    # 注：去掉 action_rate 课程收紧 —— 它会抑制 place 所需的大幅搬运动作
    # / removed action_rate curriculum: it suppresses the large motions needed for placing


############################## 事件 / Events（域随机化） ##############################
@configclass
class EventCfg:
    """重置与域随机化 / Reset & domain randomization。"""

    # startup：摩擦、末端质量随机化 / friction & ee mass randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": _EVENT_CFG["static_friction"],
            "dynamic_friction_range": _EVENT_CFG["dynamic_friction"],
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=SO101_JAW_LINK_NAME),
            "mass_distribution_params": _EVENT_CFG["ee_mass_range"],
            "operation": "add",
        },
    )

    # reset：整体重置 + cube 位姿随机 / full reset + cube pose randomization
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


############################## 环境配置 / Env config ##############################
@configclass
class So101PlaceEnvCfg(ManagerBasedRLEnvCfg):
    """SO101 抓取→放置训练环境（基础，robot/cube 由变体填充）。"""

    scene: So101PlaceSceneCfg = So101PlaceSceneCfg(
        num_envs=_SCENE_CFG["num_envs"],
        env_spacing=_SCENE_CFG["env_spacing"],
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 禁用 DLSS（分辨率 224 低于 DLSS 最小要求 300，非 headless 下会导致渲染失败）/ disable DLSS
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
