"""SO101 展示 demo，仅可视化不控制。"""
from utils.simulator import MujocoSimulator


def main():
    """启动仿真器并等待窗口关闭。"""
    sim = MujocoSimulator("config.yaml")
    sim.start()
    sim.wait()


if __name__ == "__main__":
    main()
