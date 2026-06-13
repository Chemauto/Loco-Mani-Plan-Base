# ACT Pipeline

基于 lerobot 的 ACT（Action Chunking Transformer）数据采集、训练、推理流程。

## 依赖

```bash
pip install -e /path/to/lerobot   # lerobot 源码安装
```

## 使用

```bash
python collect_data.py   # 键盘遥操作录制，Space 开始/停止，Enter 保存，Backspace 丢弃
python train_act.py      # 训练 ACT 策略
python eval_act.py       # 加载模型闭环推理
```

## 配置

所有参数在 `config.yaml` 中统一管理。
