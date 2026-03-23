# tennis_tracking (精简保留版)

这个目录是为以下 3 个功能整理后的最小可用版本：

1. 场地线检测
2. 人体骨骼关键点检测（含双人过滤）
3. 发球轨迹与击球/落地点事件分析

## 目录结构

- `videos/`：输入视频（建议命名为 `001.mp4`、`002.mp4`）
- `court_configurations/`：球场参考模板图
- `rtmlib/`：姿态估计依赖代码
- `Models/`：TrackNet 模型定义
- `WeightsTracknet/`：TrackNet 权重（`model.1`）
- `court_detector.py`：场地线检测核心
- `court_reference.py`：球场参考模板与配置
- `output/`：输出目录（可自定义）
  - `line/`：场地线结果（`{id}.npy`）
  - `pose_keypoints/`：人体关键点数据（`pose_dump_all_frames.npz`、`players_only.npy` 等）
  - `pose_video/`：骨骼关键点可视化视频（`pose_vis.mp4`、`players_only_vis.mp4`）
  - `ball/`：事件检测结果（hit/bounce）
  - `tracknetv4/`：球 2D 轨迹 CSV（可替换为你自己的路径）
- `estimate_pose.py`：人体关键点提取
- `pose_filter.py`：从多人中筛出上下两位球员
- `ball_trajectory.py`：轨迹分析工具
- `test/hit_bounce.py`：击球/落地点检测脚本

## 数据格式（按 datafomat.txt 对齐）

### 1) 球 2D 轨迹 CSV
- 典型路径：`output/tracknetv4/{id}_predict_ball.csv`
- 字段：`Frame,Visibility,X,Y`
- 说明：`Visibility=0` 时通常 `X,Y=-1,-1`

### 2) 场地线结果
- 路径：`output/line/{id}.npy`
- 形状：`(T,40)`，每帧 20 个点（x,y）
- 常用索引：`nearL=2, nearR=9, farL=6, farR=8`

### 3) 人体关键点
- 路径：`output/pose_keypoints/players_only.npy`
- 形状：`(T,2,17,2)`
- 顺序：`players[t,0]=TOP`，`players[t,1]=BOTTOM`

### 4) 事件输出
- 路径：`output/ball/*.csv`
- 关键字段：`hit`, `bounce`

## 建议流程

1. 先做场地线检测，生成 `output/line/{id}.npy`
2. 再做姿态估计 + 球员过滤，生成 `output/pose_keypoints/players_only.npy`
3. 准备球轨迹 CSV（路径可自定）
4. 运行 `test/hit_bounce.py` 做事件定位

## 当前已知样例（25fps）

- video 001：hit=197，bounce=206
- video 002：hit=55，bounce=66

## 说明

- `output` 目录名和子目录可按你习惯调整。
- 如果改了输出路径，记得同步修改脚本参数或默认路径。