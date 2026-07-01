#!/usr/bin/env python3
"""ROS2 关节状态发布：运行 MuJoCo 仿真并将关节角度发布到 /joint_states。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from utils.simulator import MujocoSimulator

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class JointStatePublisher(Node):

    def __init__(self):
        super().__init__("mujoco_joint_publisher")
        self._pub = self.create_publisher(JointState, "/joint_states", 10)
        self._sim = MujocoSimulator("config.yaml")
        self._sim.start()
        self._timer = self.create_timer(0.02, self._publish)

    def _publish(self):
        self._sim.locker.acquire()
        qpos = self._sim.data.qpos[: self._sim.model.nu].copy()
        self._sim.locker.release()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = qpos.tolist()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._sim.wait()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
