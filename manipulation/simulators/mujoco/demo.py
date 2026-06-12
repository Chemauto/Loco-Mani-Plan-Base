"""Load SO101 in MuJoCo and visualize."""
import time
from threading import Thread, Lock

import mujoco
import mujoco.viewer
import yaml


def sim_loop(model, data, viewer, locker):
    """按模型时间步长运行物理仿真。"""
    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()
        mujoco.mj_step(model, data)
        locker.release()
        time_until_next = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)


def render_loop(viewer, locker, fps):
    """按目标帧率刷新渲染，独立于仿真线程。"""
    dt = 1.0 / fps
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(dt)


def main():
    """加载配置，启动仿真和渲染线程。分线程避免大网格模型渲染阻塞仿真。"""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(cfg["robot"]["model_path"])
    data = mujoco.MjData(model)
    locker = Lock()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_thread = Thread(target=sim_loop, args=(model, data, viewer, locker))
        render_thread = Thread(target=render_loop, args=(viewer, locker, cfg["sim"]["render_fps"]))
        sim_thread.start()
        render_thread.start()
        sim_thread.join()
        render_thread.join()


if __name__ == "__main__":
    main()
