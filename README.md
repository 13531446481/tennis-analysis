# 网球分析系统 (Tennis Analysis System)

一个完整的网球视频分析工具链，用于自动检测球路、识别运动员姿态、分析击球与落地事件。

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
      - [方法 A：使用 pip (推荐)](#方法-a使用-pip-推荐)
      - [方法 B：使用 Conda](#方法-b使用-conda)
      - [验证安装](#验证安装)
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

该工具链分为 **3 大模块**：

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **球检测** | 逐帧定位网球位置 | MP4 视频 | `{id}_predict_ball.csv` |
| **姿态估计** | 识别多人17点骨骼关键点 | MP4 视频 | `pose_dump_all_frames.npz` |
| **事件分析** | 识别击球/落地帧 | CSV + NPZ | `hit_bounce.csv` |

### 使用场景

- ✅ 自动化网球比赛分析
- ✅ 骨骼姿态研究
- ✅ 击球轨迹性能评估
- ✅ 数据驱动的教练反馈

---

## 技术栈

### 核心库

- **[TrackNet V4](https://www.notion.so/TrackNet-V4-Paper)** - 球检测模型 (PyTorch)
- **[RTMLib](https://github.com/IDEA-Research/RTMLib)** - 姿态估计库 (ONNX Runtime)
- **[OpenCV](https://opencv.org/)** - 视频处理
- **[NumPy / Pandas](https://numpy.org/)** - 数据处理
- **[Scikit-learn](https://scikit-learn.org/)** - 几何计算

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

在 `videos/` 目录放入测试视频：

```bash
videos/
├── 001.mp4  # 可选
└── 002.mp4  # 可选
```

#### 2. 模型权重

**TrackNet V4 权重** (129.84MB) 不包含在仓库中，需手动下载：

```bash
# 创建权重目录
mkdir -p checkpoints/

# 下载权重到该路径
# checkpoints/tracknet-v4_best-model.pth
```

> ⚠️ 如果缺少 `.pth` 文件，`predict.py` 会报错。

#### 3. Python 环境

推荐 Python 3.8+ with CUDA 12.1+：

```bash
# 查看 Python 版本
python --version

# 验证 GPU 支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 环境配置

#### 方法 A：使用 pip (推荐)

```bash
# 1. 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 GPU 版 PyTorch（如需 CUDA）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 方法 B：使用 Conda

```bash
conda create -n tennis-analysis python=3.8
conda activate tennis-analysis
pip install -r requirements.txt
```

#### 验证安装

```bash
python -c "
import cv2, numpy, torch, rtmlib
print('✓ 环境配置完成')
print('GPU 可用:', torch.cuda.is_available())
"
```

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
[5] 事件分析 (test/hit_bounce.py)
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
├── 001_predict_ball.csv      # 球坐标序列
└── 001_predict_ball.mp4      # 可视化视频 (可选)
```

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
python estimate_pose.py \
  --device cuda \
  --backend onnxruntime \
  --mode performance

# 或仅用 CPU
python estimate_pose.py --device cpu
```

**参数：**

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--device` | 运行设备 | `cuda` / `cpu` / `mps` |
| `--backend` | 推理引擎 | `opencv` / `onnxruntime` / `openvino` |
| `--mode` | 推理模式 | `lightweight` / `balanced` / `performance` |

> **📝 性能提示：**
> - GPU (RTX 5060): ~2-5 分钟/154 帧视频
> - CPU: ~30+ 分钟
> - 用 `mode=lightweight` 可加速 30-50%

**输出：**

```
output/pose_keypoints/
├── pose_dump_all_frames.npz   # 完整数据（关键点+分数+元数据）
└── pose_keypoints_only.npy    # 仅关键点数组
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
└── players_only.npy           # 筛选后的2人关键点 (T,2,17,2)

output/pose_video/
└── players_only_vis.mp4       # 可视化视频
```

### 轨迹分析 (Trajectory Analysis)

分析球轨迹，识别击球和落地帧。

**命令：**

```bash
python ball_trajectory.py

# 或查看详细事件分析
python test/hit_bounce.py
```

**输出事件：**

```csv
Frame,Event,Height,Velocity
197,hit,2.45,25.3
206,bounce,0.15,8.2
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
| `pose_dump_all_frames.npz` | 对象数组 | 完整元数据：keypoints, scores, fps, resolution |
| `pose_keypoints_only.npy` | (T, P, 17, 2) | T=帧数, P=人数(可变) |
| `players_only.npy` | (T, 2, 17, 2) | T=帧数, 2=双人, 17=关键点, 2=坐标 |

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

路径: `output/ball/events.csv` 或 `test/hit_bounce.csv`

```csv
Frame,EventType,Description
197,hit,击球帧
206,bounce,落地帧
```

---

## 已验证数据

### 测试视频

| 视频 | 分辨率 | 帧数 | 帧率 | 已验证 |
|------|--------|------|------|--------|
| `001.mp4` | 1280×720 | 154 | 25fps | ✅ |
| `002.mp4` | 1280×720 | 154 | 25fps | ✅ |

### 已知事件

| 视频 | 击球帧 | 落地帧 | 备注 |
|------|--------|--------|------|
| 001 | 197 | 206 | 标准发球 |
| 002 | 55 | 66 | - |

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
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

如果返回 `False`，重新安装 GPU 版本：

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 可以用 CPU 跑吗？

**A:** 可以，但速度较慢（20-30 倍）。修改脚本中的 `device="cpu"`

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
├── ball_trajectory.py              # 轨迹分析
├── draw.py                         # 绘制工具
├── pose_filter.py                  # 姿态过滤
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
│   ├── pose_keypoints/             # 骨骼关键点数据
│   └── pose_video/                 # 可视化视频
│
├── test/
│   ├── hit_bounce.py               # 击球/落地分析
│   ├── ball_vs_head_report.py      # 球vs头部报告
│   └── overlay_ball_head_debug_video.py
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
