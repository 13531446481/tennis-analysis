# tracknet-series-pytorch (最小预测版)

这个目录是从原始工程中提取出来的“只保留预测功能”的最小版本。

## 1. 目录说明

- `predict.py`：预测入口脚本（已支持 `--device auto/cpu/cuda`）
- `model/tracknet_v4.py`：TrackNet V4 模型结构
- `checkpoints/tracknet-v4_best-model.pth`：预训练权重（需手动下载放入）
- `requirements.txt`：依赖列表

## 2. 权重文件说明（重要）

本仓库不再提交大模型权重（GitHub 单文件限制 100MB）。

请先准备权重文件并放到：

```text
checkpoints/tracknet-v4_best-model.pth
```

如果缺少该文件，`predict.py` 会报错：

```text
ValueError: Model weights file not found
```

## 3. 运行环境

推荐使用你当前可用的 py38 环境：

```bash
/home/fwh/miniconda3/envs/py38/bin/python
```

> 说明：你当前机器上旧版 PyTorch 与显卡 CUDA 架构有兼容警告，建议先用 CPU 跑通（`--device cpu`）。

## 3. 最小可运行命令（只导出球点 CSV）

在项目根目录执行：

```bash
python predict.py \
  --video_path videos/002.mp4 \
  --model_path checkpoints/tracknet-v4_best-model.pth \
  --only_csv \
  --device cpu
```

输出文件：

```text
/home/fwh/桌面/tenins_new/output/ball/002_predict_ball.csv
```

CSV字段：

- `Frame`
- `Visibility`
- `X`
- `Y`

## 4. 可选：输出可视化视频

去掉 `--only_csv` 即可输出预测视频（画点轨迹）：

```bash
python predict.py \
  --video_path videos/002.mp4 \
  --model_path checkpoints/tracknet-v4_best-model.pth \
  --device cpu
```

> 默认输出目录是 `output/ball`，也可通过 `--output_dir` 覆盖。

## 5. 备注

`predict.py` 当前仅依赖：

1. `model/tracknet_v4.py`
2. `checkpoints/tracknet-v4_best-model.pth`
3. 输入视频（例如 `videos/002.mp4`）

## 6. 已验证状态

已在本机完成一次最小功能验证：

- 输入：`videos/002.mp4`
- 运行：`predict.py --only_csv --device cpu`
- 输出：`002_predict_ball.csv`（成功生成）
