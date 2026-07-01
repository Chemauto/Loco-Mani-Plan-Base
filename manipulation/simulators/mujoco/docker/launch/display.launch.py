"""启动 MuJoCo 仿真 + robot_state_publisher + rviz2。"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.realpath(__file__))
_ws_root = os.path.abspath(os.path.join(_launch_dir, "../../../../.."))
_mujoco_dir = os.path.join(_ws_root, "manipulation/simulators/mujoco")
_urdf_path = os.path.join(_ws_root, "robots/assets/So101/urdf/so101_new_calib.urdf")
_mesh_dir = os.path.join(_ws_root, "robots/assets/So101/meshes")

with open(_urdf_path) as f:
    _urdf = f.read().replace("../meshes/", "file://" + _mesh_dir + "/")


def generate_launch_description():
    mujoco_node = ExecuteProcess(
        cmd=["python3", "docker/ros_demo.py"],
        cwd=_mujoco_dir,
        output="screen",
    )

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": _urdf}],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(_launch_dir, "display.rviz")],
        output="screen",
    )

    return LaunchDescription([mujoco_node, rsp_node, rviz_node])
