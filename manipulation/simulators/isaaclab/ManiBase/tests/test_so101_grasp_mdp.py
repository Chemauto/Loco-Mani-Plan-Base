from pathlib import Path
import importlib.util
import unittest
from types import SimpleNamespace

import torch


TASK_DIR = Path(__file__).resolve().parents[1] / "source" / "ManiBase" / "ManiBase" / "tasks" / "manager_based" / "so101_grasp"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, TASK_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntityCfg:
    def __init__(self, name):
        self.name = name


class SensorEntityCfg:
    """SceneEntityCfg stub exposing resolved .name and .body_ids (int or list)."""

    def __init__(self, name, body_ids=0):
        self.name = name
        self.body_ids = body_ids


class FakeScene:
    """Scene stub supporting item access (env.scene["cube"]) and .sensors."""

    def __init__(self, items, sensors):
        self._items = items
        self.sensors = sensors

    def __getitem__(self, key):
        return self._items[key]


def make_contact_env(net_forces_w):
    """Fake env with cube/ee_frame/robot (as make_env) plus a jaw contact sensor."""
    base = make_env()
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=net_forces_w))
    scene = FakeScene(items=dict(base.scene), sensors={"jaw_contact_forces": sensor})
    return SimpleNamespace(num_envs=base.num_envs, device=base.device, scene=scene)


def make_env():
    cube = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.20, 0.10, 0.08], [0.10, 0.00, 0.34]])))
    ee_frame = SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.tensor([[[0.23, 0.10, 0.08]], [[0.10, 0.00, 0.24]]])))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["base", "gripper"],
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.02], [0.23, 0.10, 0.08]], [[0.0, 0.0, 0.02], [0.10, 0.00, 0.24]]]),
        )
    )
    return SimpleNamespace(num_envs=2, device="cpu", scene={"cube": cube, "ee_frame": ee_frame, "robot": robot})


class So101GraspMdpTest(unittest.TestCase):
    def test_object_ee_distance_returns_distance_from_cube_to_target_frame(self):
        rewards = load_module("so101_rewards", "mdp/rewards.py")

        distance = rewards.object_ee_distance(make_env(), EntityCfg("ee_frame"), EntityCfg("cube"))

        self.assertTrue(torch.allclose(distance, torch.tensor([0.03, 0.10]), atol=1e-6))

    def test_object_lift_height_clamps_height_above_robot_base(self):
        rewards = load_module("so101_rewards", "mdp/rewards.py")

        height = rewards.object_lift_height(make_env(), EntityCfg("cube"), EntityCfg("robot"), robot_base_name="base")

        self.assertTrue(torch.allclose(height, torch.tensor([0.06, 0.32]), atol=1e-6))

    def test_object_height_above_base_marks_success_when_cube_is_high_enough(self):
        terminations = load_module("so101_terminations", "mdp/terminations.py")

        done = terminations.object_height_above_base(
            make_env(), EntityCfg("cube"), EntityCfg("robot"), robot_base_name="base", height_threshold=0.20
        )

        self.assertTrue(torch.equal(done, torch.tensor([False, True])))

    def test_object_position_in_robot_root_frame_returns_relative_position(self):
        observations = load_module("so101_observations", "mdp/observations.py")

        rel_pos = observations.object_position_in_robot_root_frame(make_env(), EntityCfg("robot"), EntityCfg("cube"))

        expected = torch.tensor([[0.20, 0.10, 0.06], [0.10, 0.00, 0.32]])
        self.assertTrue(torch.allclose(rel_pos, expected, atol=1e-6))

    def test_ee_to_object_vector_points_from_ee_to_cube(self):
        observations = load_module("so101_observations", "mdp/observations.py")
        env = make_contact_env(torch.zeros(2, 1, 3))

        vec = observations.ee_to_object_vector(env, EntityCfg("ee_frame"), EntityCfg("cube"))

        expected = torch.tensor([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.10]])
        self.assertTrue(torch.allclose(vec, expected, atol=1e-6))

    def test_gripper_object_contact_flags_force_above_threshold(self):
        rewards = load_module("so101_rewards", "mdp/rewards.py")
        # env0: 0.5N (below 1.0), env1: 2.0N (above)
        forces = torch.tensor([[[0.0, 0.0, 0.5]], [[0.0, 0.0, 2.0]]])  # (2, 1, 3)
        env = make_contact_env(forces)

        contact = rewards.gripper_object_contact(env, SensorEntityCfg("jaw_contact_forces", body_ids=0), threshold=1.0)

        self.assertTrue(torch.equal(contact, torch.tensor([0.0, 1.0])))

    def test_gripper_object_contact_handles_list_body_ids(self):
        rewards = load_module("so101_rewards", "mdp/rewards.py")
        forces = torch.tensor([[[0.0, 0.0, 0.5]], [[0.0, 0.0, 2.0]]])  # (2, 1, 3)
        env = make_contact_env(forces)

        contact = rewards.gripper_object_contact(env, SensorEntityCfg("jaw_contact_forces", body_ids=[0]), threshold=1.0)

        self.assertTrue(torch.equal(contact, torch.tensor([0.0, 1.0])))

    def test_object_lift_height_when_grasped_gates_by_contact(self):
        rewards = load_module("so101_rewards", "mdp/rewards.py")
        # env0 not grasping (0.5N), env1 grasping (2.0N); base lift heights are [0.06, 0.32]
        forces = torch.tensor([[[0.0, 0.0, 0.5]], [[0.0, 0.0, 2.0]]])
        env = make_contact_env(forces)

        lifted = rewards.object_lift_height_when_grasped(
            env,
            EntityCfg("cube"),
            EntityCfg("robot"),
            sensor_cfg=SensorEntityCfg("jaw_contact_forces", body_ids=0),
            robot_base_name="base",
            threshold=1.0,
        )

        # env0 gated to 0 (not grasping), env1 keeps 0.32
        self.assertTrue(torch.allclose(lifted, torch.tensor([0.0, 0.32]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
