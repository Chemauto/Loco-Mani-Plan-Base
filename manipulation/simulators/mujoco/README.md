# MuJoCo 仿真

SO101 机械臂的 MuJoCo 仿真环境。

```bash
python demo_show.py      # 纯展示
python demo_control.py   # 键盘控制，数字键正转/字母键反转六个关节
```

## RViz 可视化（Docker）

```bash
# 1. 构建镜像（仅首次）
cd docker && ./build.sh

# 2. 进入容器
./run.sh

# 容器内一键启动仿真 + robot_state_publisher + rviz2:
ros2 launch docker/launch/display.launch.py
```
