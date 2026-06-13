"""SO101 机器人资产配置。"""

from pathlib import Path

import isaaclab.sim as sim_utils
import yaml
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SO101_ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_GRIPPER_JOINT_NAME = "gripper"
SO101_BASE_LINK_NAME = "base_link"
SO101_EE_LINK_NAME = "gripper_frame_link"
SO101_JAW_LINK_NAME = "moving_jaw_so101_v1_link"

_REPO_ROOT = Path(__file__).resolve().parents[8]
_ISAACLAB_DIR = _REPO_ROOT / "manipulation" / "simulators" / "isaaclab"
_CONFIG_PATH = _ISAACLAB_DIR / "config.yaml"
_DEFAULT_URDF_PATH = "../../../robots/assets/So101/urdf/so101_new_calib.urdf"


def _resolve_urdf_path() -> str:
    cfg = yaml.safe_load(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    robot_cfg = cfg.get("so101_grasp", {}).get("robot", {})
    rel_path = robot_cfg.get("urdf_path", _DEFAULT_URDF_PATH)
    return str((_ISAACLAB_DIR / rel_path).resolve())


SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_resolve_urdf_path(),
        fix_base=True,
        root_link_name=SO101_BASE_LINK_NAME,
        merge_fixed_joints=False,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=35.0, damping=4.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.25, 0.26),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.6,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "sts3215_arm": ImplicitActuatorCfg(
            joint_names_expr=SO101_ARM_JOINT_NAMES,
            effort_limit_sim=3.35,
            velocity_limit_sim=2.5,
            stiffness=35.0,
            damping=4.0,
        ),
        "sts3215_gripper": ImplicitActuatorCfg(
            joint_names_expr=[SO101_GRIPPER_JOINT_NAME],
            effort_limit_sim=3.35,
            velocity_limit_sim=2.5,
            stiffness=35.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
