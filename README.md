# tennis-analysis

## Installation

- follow the instructions in [rtmlib](https://github.com/Tau-J/rtmlib#installation).

### install rtmlib

- install from pypi:

```shell
pip install rtmlib -i https://pypi.org/simple
```

- install from source code:

```shell
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib

pip install -r requirements.txt

pip install -e .

# [optional]
# pip install onnxruntime-gpu
# pip install openvino

```

### clone tennis_tracking

```shell
git clone https://github.com/ArtLabss/tennis-tracking.git
```

### install requirements

```shell
pip install -r requirements.txt
```

### Structure

- clips.py: 将视频切分成多个片段
- tennis_detector.py 识别视频片段中的网球和球场线
- pose_estimation_rtmlib.py 识别人体姿势
- ball_trajectory.py 识别视频片段中发球的轨迹
- demo.py: gradio可视化，功能演示
- dataloader.py/loss.py/pose_predict_model.py/train.py 姿势预测模型和训练代码

### Todo

- 两边不同运动员的发球
- 放到同一个函数里识别
- 将速度输出成一个文件

## Acknowledgement

Our code is based on these repos:

- [tennis-tracking](https://github.com/ArtLabss/tennis-tracking)
- [rtmlib](https://github.com/Tau-J/rtmlib#installation)
