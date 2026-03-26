# 网球分析系统 (Tennis Analysis System)

一个面向网球发球分析的完整工具链，用于自动检测网球、识别球员姿态、分析击球/落地事件，并反算发球初速度。

> 本项目整合了 TrackNet V4 球检测、RTMLib 姿态估计、以及自定义事件检测算法。

---

## 📋 目录

- [网球分析系统 (Tennis Analysis System)](#网球分析系统-tennis-analysis-system)
  - [📋 目录](#-目录)
  - [关于本项目](#关于本项目)
    - [核心功能](#核心功能)
    - [使用场景](#使用场景)
  - [技术栈](#技术栈)
    - [核心库](#核心库)
    - [硬件要求](#硬件要求)
  - [快速开始](#快速开始)
    - [前置条件](#前置条件)
      - [1. 视频文件](#1-视频文件)
      - [2. 模型权重](#2-模型权重)
      - [3. Python 环境](#3-python-环境)
    - [环境配置](#环境配置)
      - [使用 Conda（推荐）](#使用-conda推荐)
      - [验证安装](#验证安装)
      - [依赖库与配置说明](#依赖库与配置说明)
  - [使用指南](#使用指南)
    - [完整流程](#完整流程)
    - [球检测 (Ball Detection)](#球检测-ball-detection)
    - [姿态估计 (Pose Estimation)](#姿态估计-pose-estimation)
    - [双人过滤 (Player Filtering)](#双人过滤-player-filtering)
    - [轨迹分析 (Trajectory Analysis)](#轨迹分析-trajectory-analysis)
  - [输出格式](#输出格式)
    - [1. 球检测 CSV](#1-球检测-csv)
    - [2. 场地线 NPY](#2-场地线-npy)
    - [3. 骨骼关键点 NPY/NPZ](#3-骨骼关键点-npynpz)
    - [4. 事件输出 CSV](#4-事件输出-csv)
  - [已验证数据](#已验证数据)
    - [测试视频](#测试视频)
    - [已知事件](#已知事件)
  - [常见问题](#常见问题)
    - [Q1: 球检测输出为什么是 153 行而不是 154 行？](#q1-球检测输出为什么是-153-行而不是-154-行)
    - [Q2: 显卡不被识别怎么办？](#q2-显卡不被识别怎么办)
    - [Q3: 可以用 CPU 跑吗？](#q3-可以用-cpu-跑吗)
    - [Q4: 如何处理自定义视频？](#q4-如何处理自定义视频)
    - [Q5: 输出目录可以自定义吗？](#q5-输出目录可以自定义吗)
  - [目录结构](#目录结构)
  - [许可证](#许可证)
  - [反馈与建议](#反馈与建议)
  - [致谢](#致谢)
    - [核心依赖库](#核心依赖库)
    - [参考项目与依托工程](#参考项目与依托工程)

---

## 关于本项目

### 核心功能

该工具链目前分为 **5 个核心阶段**：

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **球检测** | 逐帧定位网球位置 | MP4 视频 | `{id}_predict_ball.csv` |
| **场地线检测** | 提取标准球场线点 | MP4 视频 | `{id}.npy` |
| **姿态估计 + 双人过滤** | 输出双人关键点 | MP4 视频 + line | `2_keypoints.npy` |
| **事件分析** | 识别击球/落地帧 | ball CSV + players NPY | `hit_bounce.csv` |
| **速度求解** | 反算发球初速度 | hit/bounce + line + players | `step2_velocity.json` |

### 使用场景

- ✅ 自动化网球比赛分析
- ✅ 骨骼姿态研究
- ✅ 发球速度反算与可视化演示
- ✅ 数据驱动的教练反馈

---

## 技术栈

### 核心库

- **[TrackNet V4](https://www.notion.so/TrackNet-V4-Paper)** - 球检测模型 (PyTorch)
- **[RTMLib](https://github.com/IDEA-Research/RTMLib)** - 姿态估计库 (ONNX Runtime)
- **[OpenCV](https://opencv.org/)** - 视频处理
- **[NumPy / Pandas](https://numpy.org/)** - 数据处理
- **[Gradio](https://gradio.app/)** - 交互式演示界面

### 硬件要求

| 硬件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | 任何 NVIDIA GPU (≥2GB VRAM) | RTX 3060+ |
| CPU | Intel i5/AMD Ryzen 5 | i7/Ryzen 7 |
| 内存 | 8GB | 16GB |

---

## 快速开始

### 前置条件

#### 1. 视频文件

仓库当前提供了 `001` 到 `014` 的测试视频，直接放在 `videos/` 目录：

```bash
videos/
├── 001.mp4
├── 002.mp4
├── 003.mp4
├── ...
└── 014.mp4
```

#### 2. 模型权重

**TrackNet V4 权重** 不包含在仓库中，需手动下载到 `checkpoints/tracknet-v4_best-model.pth`。

参考来源：

- Releases 页面：`https://github.com/AnInsomniacy/tracknet-series-pytorch/releases`
- 直接下载链接：`https://github.com/AnInsomniacy/tracknet-series-pytorch/releases/latest/download/tracknet-v4_best-model.pth`

```bash
# 创建权重目录
mkdir -p checkpoints/

# 下载权重到该路径
wget -O checkpoints/tracknet-v4_best-model.pth \
  https://github.com/AnInsomniacy/tracknet-series-pytorch/releases/latest/download/tracknet-v4_best-model.pth
```

> ⚠️ 如果缺少 `.pth` 文件，`predict.py` 会报错。

#### 3. Python 环境

推荐环境（已验证）：Python 3.10 + CUDA 12.8 + PyTorch 2.11.0+cu128（RTX 50 系列可用）。

```bash
# 查看 Python 版本
python --version

# 验证 GPU 支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 环境配置

#### 使用 Conda（推荐）

```bash
# 1) 创建并激活环境
conda create -n py310 python=3.10 -y
conda activate py310

# 2) 安装 GPU 版 PyTorch（CUDA 12.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3) 安装项目基础依赖
pip install -r requirements.txt

# 4) 安装完整流程建议依赖（用于轨迹图/姿态后处理）
pip install matplotlib scikit-learn onnxruntime-gpu
```

#### 验证安装

```bash
python -c "
import cv2, numpy, torch
print('✓ 环境配置完成')
print('GPU 可用:', torch.cuda.is_available())
print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```

#### 依赖库与配置说明

| 库 | 用途 | 是否必需 | 配置方式 |
|------|------|------|------|
| `torch`, `torchvision`, `torchaudio` | TrackNet V4 推理（`predict.py`） | 必需 | 使用 `cu128` 源安装 GPU 版 |
| `opencv-python` | 视频读写、绘制与保存 | 必需 | 已在 `requirements.txt` |
| `numpy` | 数值计算与数组处理 | 必需 | 已在 `requirements.txt` |
| `pandas` | 球检测 CSV 读写 | 必需 | 已在 `requirements.txt` |
| `scipy` | 场地线/信号处理 | 必需 | 已在 `requirements.txt` |
| `tqdm` | 进度条显示 | 必需 | 已在 `requirements.txt` |
| `pyyaml` | 配置读取 | 必需 | 已在 `requirements.txt` |
| `tensorboard` | 训练/调试日志 | 可选 | 已在 `requirements.txt` |
| `matplotlib` | 轨迹图与调试可视化 | 建议安装 | `pip install matplotlib` |
| `scikit-learn` | 轨迹去噪（如 LOF） | 建议安装 | `pip install scikit-learn` |
| `onnxruntime-gpu` | RTMLib 的 GPU 后端 | 姿态模块推荐 | `pip install onnxruntime-gpu` |

建议：
- 只跑球检测（`predict.py`）：安装 PyTorch + `requirements.txt` 即可。
- 跑完整流程（球检测 + 姿态 + 轨迹分析）：额外安装 `matplotlib`、`scikit-learn`、`onnxruntime-gpu`。

---

## 使用指南

### 完整流程

标准的端到端流程：

```
视频输入
  ↓
[1] 球检测 (predict.py)
  ↓
[2] 场地线检测 (court_detector.py)
  ↓
[3] 姿态估计 (estimate_pose.py)
  ↓
[4] 双人过滤 (pose_filter.py)
  ↓
[5] 事件分析 (hit_bounce.py)
  ↓
分析结果 (CSV/视频)
```

### 球检测 (Ball Detection)

使用 TrackNet V4 检测网球。

**命令：**

```bash
# 仅输出 CSV
python predict.py \
  --video_path videos/001.mp4 \
  --model_path checkpoints/tracknet-v4_best-model.pth \
  --only_csv \
  --device cuda

# 输出 CSV + 可视化视频
python predict.py \
  --video_path videos/001.mp4 \
  --device cuda
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video_path` | 输入视频路径 | `videos/001.mp4` |
| `--model_path` | 模型权重路径 | `checkpoints/tracknet-v4_best-model.pth` |
| `--only_csv` | 仅输出CSV（跳过视频生成） | False |
| `--device` | 运行设备 (cuda/cpu) | `cuda` |
| `--output_dir` | 输出目录 | `output/ball/` |

**输出：**

```
output/ball/
└── 001_predict_ball.csv      # 球坐标序列（默认/推荐输出）
```

说明：

- 当前项目常用流程默认使用 `--only_csv`，因此通常只生成 `output/ball/{id}_predict_ball.csv`
- 如果运行 `predict.py` 时不带 `--only_csv`，还会额外生成可视化视频：`output/ball/{id}_predict.mp4`

**CSV 格式：**

```csv
Frame,Visibility,X,Y
0,1,640.2,360.5
1,1,641.1,359.8
2,0,-1,-1
```

### 姿态估计 (Pose Estimation)

提取每帧所有人物的 17 个关键点。

**命令：**

```bash
python -c "from estimate_pose import dump_pose_from_video; dump_pose_from_video(video_path='videos/001.mp4', out_dir='output/pose_keypoints/001', device='cuda', backend='onnxruntime', mode='performance', to_openpose=False, max_frames=-1)"
```

**参数：**

常用参数：

- `device`: `cuda` / `cpu`
- `backend`: `onnxruntime` / `opencv` / `openvino`
- `mode`: `lightweight` / `balanced` / `performance`
- `max_frames=-1`: 处理完整视频

> **📝 性能提示：**
> - GPU (RTX 5060): ~2-5 分钟/154 帧视频
> - CPU: ~30+ 分钟
> - 用 `mode=lightweight` 可加速 30-50%

**输出：**

```
output/pose_keypoints/001/
├── dump.npz            # 完整数据（关键点+分数+元数据）
└── multi_keypoints.npy # 多人关键点数组
```

### 双人过滤 (Player Filtering)

从多人检测结果中自动选出上下两位球员。

**命令：**

```bash
python pose_filter.py
```

**工作原理：**

1. 从 `output/line/001.npy` 读取场地基线
2. 计算每个人到基线的距离
3. 选择距离最近的 2 人（通常为对打的两位）
4. 生成可视化视频

**输出：**

```
output/pose_keypoints/
└── 2_keypoints.npy     # 筛选后的2人关键点 (T,2,17,2)

output/pose_video/
└── 001_players.mp4     # 可视化视频
```

### 事件分析与速度求解 (Event + Speed)

分析球轨迹，识别击球和落地帧，并支持预处理轨迹可视化核对。

**击球高度建模说明（用于 3D 初速度反解）：**

- 仅用 `1.85m`（身高）会低估击球点高度。
- 当前默认将击球高度设为 `1.85 * 1.5 = 2.775m`，用于近似“身高 + 手臂伸展”后的击球高度。
- 对应脚本默认参数：
  - `step2_initial_velocity.py` 中 `--z_hit=2.775`
  - `run_full_pipeline.py` 中 `--z_hit=2.775`
- 若视频中击球姿态差异较大，可按样本手动调整 `--z_hit`。

**命令：**

```bash
# 事件分析
python hit_bounce.py --video videos/001.mp4 --players output/pose_keypoints/001/2_keypoints.npy

# 二维映射
python step1_standard_2d.py \
  --video_id 001 \
  --hit_frame 197 \
  --bounce_frame 206 \
  --server_id 1 \
  --video_path videos/001.mp4 \
  --line_npy output/line/001.npy \
  --players_npy output/pose_keypoints/001/2_keypoints.npy \
  --ball_csv output/ball/001_predict_ball.csv

# 速度求解
python step2_initial_velocity.py \
  --video_id 001 \
  --step1_json output/step1_2d/001/step1_2d.json \
  --z_hit 2.775 \
  --z_bounce 0.0
```

**相关脚本：**

- `hit_bounce.py`：主入口（自动输出 hit/bounce）
- `step1_standard_2d.py`：击球/落地点映射到标准球场坐标系
- `step2_initial_velocity.py`：基于 2D+高度先验反算初速度
- `test/plot_line_with_events.py`：预处理轨迹与事件点可视化

**输出事件（示例）：**

```csv
video_id,hit,bounce,hit_sec,bounce_sec
001,197,205,7.88,8.2
```

---

## 输出格式

### 1. 球检测 CSV

路径: `output/ball/{id}_predict_ball.csv`

```
Frame: 帧索引 (0-based)
Visibility: 0=不可见, 1=可见
X, Y: 像素坐标 (通常为负值表示不可见)
```

### 2. 场地线 NPY

路径：`output/line/{id}.npy`

形状: `(T, 40)` → 20 个点 × (x,y)

常用索引：
- `nearL, nearR`: 近端左/右
- `farL, farR`: 远端左/右

### 3. 骨骼关键点 NPY/NPZ

路径: `output/pose_keypoints/`

| 文件 | 格式 | 说明 |
|------|------|------|
| `dump.npz` | 对象数组 | 完整元数据：keypoints, scores, fps, resolution |
| `multi_keypoints.npy` | (T, P, 17, 2) | T=帧数, P=人数(可变) |
| `2_keypoints.npy` | (T, 2, 17, 2) | T=帧数, 2=双人, 17=关键点, 2=坐标 |

**17 个关键点顺序（COCO 格式）：**

```
0: 鼻子     1: 左眼    2: 右眼
3: 左耳     4: 右耳    5: 左肩
6: 右肩     7: 左肘    8: 右肘
9: 左腕     10: 右腕   11: 左髋
12: 右髋    13: 左膝   14: 右膝
15: 左踝    16: 右踝
```

### 4. 事件输出 CSV

路径: `output/hit_bounce/{id}.csv` 或 `output/ball/hit_from_turns_{id}.csv`

```csv
video_id,hit,bounce,hit_sec,bounce_sec
001,197,205,7.88,8.2
```

说明：
- `hit_bounce.py` 默认输出 `output/hit_bounce/{id}.csv`
- `hit_bounce.py` 命令行输出包含 `toss_apex`、`reason` 等诊断信息

### Gradio 演示界面

- 启动命令：`python gradio_pipeline.py`
- 默认从 `videos/` 目录选择本地视频，按步骤完成球检测、姿态、击球落地与发球速度计算
- 主实现：`gradio/app.py`
- 界面样式：`gradio/styles.css`

---

## 已验证数据

### 测试视频

| 视频 | 分辨率 | 帧数 | 帧率 | 已验证 |
|------|--------|------|------|--------|
| `001.mp4` | 1280×720 | 333 | 25fps | ✅ |
| `002.mp4` | 1280×720 | 154 | 25fps | ✅ |
| `003` - `014` | - | - | - | 部分验证 |

### 已知事件

| 视频 | 击球帧 | 落地帧 | 备注发球速度 |
|------|--------|--------|------|
| 001 | 197 | 206 | 203 |
| 002 | 55 | 66 | 153 |
| 003 | 21 | 33 | 170 |
| 004 | 33 | 43 | 197 |？
| 005 | 29 | 42 | 172 |  ？
| 006 | 28 | 38 | 194 |？
| 007 | 44 | 57 | 158 |？
| 008 | 32 | 43 | 200 |
| 009 | 228 | 239 | 203 |
| 010 | 37 | 49 | 193 |
| 011 | 34 | 48 | 177 | 
| 012 | 10 | 22 | 188 |
| 013 | 25 | 35 | 203 |
| 014 | 155 | 168 | 148 |

---

## 常见问题

### Q1: 球检测输出为什么是 153 行而不是 154 行？

**A:** 已在 v1.0.1 修复。更新代码：

```bash
git pull origin main
```

现在 CSV 行数应与视频总帧数一致。

### Q2: 显卡不被识别怎么办？

**A:** 检查 PyTorch + CUDA 安装：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
```

如果返回 `False`，重新安装 GPU 版本：

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Q3: 可以用 CPU 跑吗？

**A:** 可以，但速度较慢。命令行脚本可显式传 `--device cpu`，Gradio 界面也可在高级设置中切换。

### Q4: 如何处理自定义视频？

**A:** 放入 `videos/` 目录，然后通过 `--video_path` 指定：

```bash
python predict.py --video_path videos/my_video.mp4
```

### Q5: 输出目录可以自定义吗？

**A:** 可以。修改脚本中的默认路径，或通过参数传入：

```bash
python predict.py --output_dir /custom/path
```

---

## 目录结构

```
.
├── README.md                       # 本文件
├── requirements.txt                # 依赖列表
│
├── predict.py                      # 球检测 (TrackNet V4)
├── court_detector.py               # 场地线检测
├── court_reference.py              # 场地参考模板
├── estimate_pose.py                # 姿态估计 (RTMLib)
├── pose_filter.py                  # 双人过滤
├── run_full_pipeline.py            # 命令行全流程入口
├── step1_standard_2d.py            # 标准球场二维映射
├── step2_initial_velocity.py       # 发球初速度求解
│
├── model/
│   └── tracknet_v4.py              # TrackNet V4 模型定义
│
├── checkpoints/
│   └── tracknet-v4_best-model.pth  # ⚠️ 需手动下载
│
├── videos/
│   ├── 001.mp4                     # 测试视频1
│   └── 002.mp4                     # 测试视频2
│
├── output/
│   ├── ball/                       # 球检测结果
│   ├── line/                       # 场地线检测结果
│   ├── hit_bounce/                 # 击球/落地结果
│   ├── pose_keypoints/             # 骨骼关键点数据
│   ├── step1_2d/                   # 标准球场坐标映射结果
│   ├── step2_velocity/             # 球速结果
│   └── step2_trajectory/           # 轨迹可视化结果
│
├── test/
│   └── plot_line_with_events.py    # 预处理轨迹可视化
│
├── hit_bounce.py                   # 击球/落地分析主入口
├── gradio_pipeline.py              # Gradio 启动别名
├── gradio/
│   ├── app.py                      # Gradio 主实现
│   └── styles.css                  # Gradio 界面样式
│
└── rtmlib/                         # RTMLib (骨骼检测库)
    ├── rtmlib/
    ├── setup.py
    └── requirements.txt
```

---

## 许可证

本项目采用 [MIT License](LICENSE)。

---

## 反馈与建议

如有问题或建议，欢迎提交 Issue：

- 🐛 [报告 Bug](https://github.com/13531446481/tennis-analysis/issues/new?labels=bug)
- 💡 [功能建议](https://github.com/13531446481/tennis-analysis/issues/new?labels=enhancement)

---

## 致谢

### 核心依赖库

- [TrackNet V4](https://github.com/yastrebksv/TrackNet) - 球检测模型
- [RTMLib](https://github.com/IDEA-Research/RTMLib) - 多人姿态估计
- [OpenCV](https://github.com/opencv/opencv) - 视频处理
- [NumPy](https://numpy.org/) - 数值计算
- [PyTorch](https://pytorch.org/) - 深度学习框架

### 参考项目与依托工程

本项目在以下优秀开源项目的基础上进行定制和优化：

- **[tennis-tracking](https://github.com/ArtLabss/tennis-tracking)** - 网球轨迹检测与事件分析算法参考
- **[tracknet-series-pytorch](https://github.com/AnInsomniacy/tracknet-series-pytorch)** - TrackNet PyTorch 实现与模型转换

感谢这些项目为网球AI分析领域做出的贡献！

---

**最后更新** © 2026 | [项目主页](https://github.com/13531446481/tennis-analysis)
