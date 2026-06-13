"""MuJoCo 仿真器，分离仿真线程和渲染线程。"""
import time
from threading import Thread, Lock

import mujoco
import mujoco.viewer
import yaml


class MujocoSimulator:
    """MuJoCo 仿真器，管理模型加载、仿真循环和渲染循环。"""

    def __init__(self, config_path="config.yaml"):
        """加载配置和模型，初始化线程锁。"""
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.model = mujoco.MjModel.from_xml_path(self.cfg["robot"]["model_path"])
        self.data = mujoco.MjData(self.model)
        self.locker = Lock()
        self.viewer = None

    def start(self):
        """启动仿真和渲染线程。"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = 1.5

        self._sim_thread = Thread(target=self._sim_loop, daemon=True)
        self._render_thread = Thread(target=self._render_loop, daemon=True)
        self._sim_thread.start()
        self._render_thread.start()

    def is_running(self):
        """检查仿真是否仍在运行。"""
        return self.viewer is not None and self.viewer.is_running()

    def _sim_loop(self):
        """按模型时间步长运行物理仿真。"""
        while self.viewer.is_running():
            step_start = time.perf_counter()
            self.locker.acquire()
            mujoco.mj_step(self.model, self.data)
            self.locker.release()
            dt = self.model.opt.timestep - (time.perf_counter() - step_start)
            if dt > 0:
                time.sleep(dt)

    def _render_loop(self):
        """按目标帧率刷新渲染，独立于仿真线程。"""
        fps = self.cfg["sim"]["render_fps"]
        dt = 1.0 / fps
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(dt)

    def wait(self):
        """阻塞等待仿真结束。"""
        if self._sim_thread.is_alive():
            self._sim_thread.join()
        if self._render_thread.is_alive():
            self._render_thread.join()
