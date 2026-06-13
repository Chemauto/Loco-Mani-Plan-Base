"""键盘遥操作录制 LeRobotDataset，采集图像 + 关节状态 + 动作。"""
import shutil
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simulators" / "mujoco"))
from utils.controller import KeyboardController
from utils.simulator import MujocoSimulator

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 热键映射
HOTKEYS = {
    " ": "toggle_recording",
    "\r": "save",
    "\n": "save",
    "\x7f": "discard",
    "\x04": "discard",
    "\x1b": "quit",
    "z": "reset",
}


class DataCollector:
    """键盘遥操作数据采集器，复用 MujocoSimulator + KeyboardController。"""

    def __init__(self, config_path="config.yaml"):
        """加载配置，初始化仿真器、控制器、渲染器和数据集。"""
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        sim_cfg_path = str(Path(config_path).parent / self.cfg["simulator_config"])
        self.sim = MujocoSimulator(sim_cfg_path)
        self.model = self.sim.model
        self.data = self.sim.data
        self.ctrl = KeyboardController(self.cfg["controller"]["joint_step"])
        self.ctrl.set_model(self.model)

        cam = self.cfg["camera"]
        self.renderer = mujoco.Renderer(self.model, cam["height"], cam["width"])
        self.camera_name = cam["name"]

        # 腕部相机
        self.wrist_cam = mujoco.MjvCamera()
        self.wrist_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.wrist_cam.trackbodyid = self.model.body("gripper").id
        self.wrist_cam.distance = 0.15
        self.wrist_cam.elevation = -90
        self.wrist_cam.azimuth = 0

        ds_cfg = self.cfg["dataset"]
        self.fps = ds_cfg["fps"]
        self.num_episodes = ds_cfg["num_episodes"]

        img_key = f"observation.images.{cam['name']}"
        self.img_key = img_key
        self.wrist_img_key = "observation.images.wrist"
        features = {
            img_key: {
                "dtype": "image",
                "shape": (cam["height"], cam["width"], 3),
                "names": ["height", "width", "channels"],
            },
            self.wrist_img_key: {
                "dtype": "image",
                "shape": (cam["height"], cam["width"], 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["action"],
            },
        }

        root = Path(config_path).parent / ds_cfg["root"]
        if root.exists():
            print(f"数据目录 {root} 已存在，删除重建")
            shutil.rmtree(root)
        self.dataset = LeRobotDataset.create(
            repo_id=ds_cfg["repo_id"],
            fps=self.fps,
            features=features,
            root=str(root),
            robot_type="so101",
            use_videos=False,
        )

        self.recording = False
        self.episode_count = 0
        self.frame_count = 0

        # 初始化 rerun
        rr.init("so101_collect", spawn=True)

    def _render(self, camera):
        """渲染指定相机的图像。"""
        self.renderer.update_scene(self.data, camera)
        return self.renderer.render()

    def _log_rerun(self, agent_img, wrist_img, state):
        """将图像和状态记录到 rerun。"""
        rr.log("camera/agentview", rr.Image(agent_img))
        rr.log("camera/wrist", rr.Image(wrist_img))
        for i in range(len(state)):
            rr.log(f"joint/j{i}", rr.Scalars(float(state[i])))
        rr.log("status/recording", rr.TextDocument("录制中" if self.recording else "暂停"))
        rr.log("status/episode", rr.TextDocument(f"Episode {self.episode_count}  帧 {self.frame_count}"))

    def _handle_hotkey(self, key):
        """处理热键动作。返回 'quit' 或 'done' 表示退出。"""
        if key == "quit":
            return "quit"

        if key == "reset":
            self.ctrl.reset()
            if self.frame_count > 0:
                self.dataset.clear_episode_buffer()
                self.frame_count = 0
            self.recording = False
            print("重置归零")

        elif key == "toggle_recording":
            self.recording = not self.recording
            if self.recording:
                print(f"[Episode {self.episode_count}] 开始录制...")
            else:
                print(f"[Episode {self.episode_count}] 暂停录制 ({self.frame_count} 帧)")

        elif key == "save":
            if self.frame_count > 0:
                self.dataset.save_episode()
                self.episode_count += 1
                self.frame_count = 0
                self.recording = False
                self.ctrl.reset()
                print(f"Episode {self.episode_count - 1} 已保存，已重置归零")
                if self.episode_count >= self.num_episodes:
                    print(f"已完成 {self.num_episodes} 个 episode，结束采集")
                    return "done"
            else:
                print("没有可保存的帧，请先按 Space 开始录制")

        elif key == "discard":
            if self.frame_count > 0:
                self.dataset.clear_episode_buffer()
                self.frame_count = 0
                self.recording = False
                self.ctrl.reset()
                print("当前 episode 已丢弃，已重置归零")

        return None

    def run(self):
        """主循环：20Hz 遥操作录制。"""
        self.sim.start()
        dt = 1.0 / self.fps
        print(f"数据采集启动，目标 {self.num_episodes} 个 episode")
        print("按键: 1-6/Q-Y 控关节 | Space 录制/暂停 | Enter 保存 | Backspace 丢弃 | Z 重置 | ESC 退出")

        try:
            while self.sim.is_running():
                t0 = time.time()

                # 锁内：更新控制器
                self.sim.locker.acquire()
                hotkeys = self.ctrl.update()
                action_cmd = self.ctrl.get_cmd().astype(np.float32)
                self.data.ctrl[:] = action_cmd
                state = self.data.qpos[:self.model.nu].copy().astype(np.float32)
                self.sim.locker.release()

                # 处理热键
                for ch in hotkeys:
                    action = HOTKEYS.get(ch)
                    if action:
                        result = self._handle_hotkey(action)
                        if result in ("quit", "done"):
                            break

                # 渲染两个相机
                agent_img = self._render(self.camera_name)
                wrist_img = self._render(self.wrist_cam)

                # rerun 可视化
                self._log_rerun(agent_img, wrist_img, state)

                # 录制帧
                if self.recording:
                    self.dataset.add_frame({
                        self.img_key: agent_img,
                        self.wrist_img_key: wrist_img,
                        "observation.state": state,
                        "action": action_cmd,
                        "task": "teleop",
                    })
                    self.frame_count += 1

                # 控制频率
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\n采集中断")

        finally:
            self.ctrl.restore()

        if self.frame_count > 0:
            self.dataset.save_episode()
            print(f"自动保存最后 episode ({self.frame_count} 帧)")

        self.dataset.finalize()
        self.sim.wait()
        print(f"采集完成，共 {self.episode_count} 个 episode")


if __name__ == "__main__":
    DataCollector().run()
