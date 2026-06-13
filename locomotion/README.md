# 🦿 Locomotion

足式与轮式机器人的运动控制，涵盖基于强化学习的步态生成、地形适应等基础训练流程。

参考项目：[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)、[Legged Gym](https://github.com/leggedrobotics/legged_gym)

## 本地安装备注

如果需要修改 `rsl_rl` 的网络结构，先使用本仓库内的本地源码版本安装：

```bash
cd LocoBase
python -m pip install -e source/my_rsl_rl
python -m pip install -e source/LocoBase
```

