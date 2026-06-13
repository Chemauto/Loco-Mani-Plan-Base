"""SO101 键盘控制 demo。按键说明：数字键正转，字母键反转。1/Q=肩平移, 2/W=肩升降, 3/E=肘, 4/R=腕俯仰, 5/T=腕旋转, 6/Y=夹爪"""
import time

from utils.simulator import MujocoSimulator
from utils.controller import KeyboardController


def main():
    """初始化仿真器和键盘控制器，轮询按键并下发关节指令。"""
    sim = MujocoSimulator("config.yaml")
    ctrl = KeyboardController(joint_step=sim.cfg["controller"]["joint_step"])
    ctrl.set_model(sim.model)
    sim.start()

    try:
        while sim.is_running():
            sim.locker.acquire()
            ctrl.update()
            sim.data.ctrl[:] = ctrl.get_cmd()
            sim.locker.release()
            time.sleep(0.02)
    finally:
        ctrl.restore()

    sim.wait()


if __name__ == "__main__":
    main()
