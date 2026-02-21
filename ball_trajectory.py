import copy
import os
from copy import deepcopy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from draw import draw_mmpose
from rtmlib.visualization.skeleton import coco17

x = 20.42
h = 3.29827828


def calculate_ball_speed(horizontal_distance, vertical_distance, total_frame_count, fps=25):
    """
    计算球的速度。

    参数:
    horizontal_distance (float): 水平距离。
    vertical_distance (float): 垂直距离。
    total_frame_count (int): 总帧数。
    fps (int): 帧率，默认为25。

    返回:
    float: 球的速度。
    """
    g = 9.8
    time = 1.0 / fps * total_frame_count
    v0_y = (vertical_distance / time - 0.5 * g * time) * 3.6
    v0_x = horizontal_distance / time * 3.6
    return np.linalg.norm([v0_x, v0_y])


def detect_kpt(video, balls):
    """
    检测视频中球的关键点。

    参数:
    video (str): 视频文件的路径。
    balls (list): 球的坐标列表。

    返回:
    无
    """
    from demo import detect_pose
    point = (balls[0][0], balls[0][1])
    file_name, _ = os.path.splitext(os.path.basename(video))
    detect_pose(video, point, output_path=f'./cuts/keypoints/{file_name}.npy')


def draw_img(video, keypoints, balls, waitkey=1):
    """
    根据给定的关键点信息在视频每一帧上绘制关键点和骨架。

    参数:
    video (str): 视频文件的路径。
    keypoints (np.ndarray): 包含关键点信息的列表，每个元素对应一帧的关键点坐标。
    waitkey (int): 图片显示的等待时间。

    返回:
    无
    """
    cap = cv2.VideoCapture(video)
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            file_name, _ = os.path.splitext(os.path.basename(video))
            img = draw_mmpose(frame, keypoints[count][0], coco17['keypoint_info'], coco17['skeleton_info'])
            head = (int(keypoints[count][0][0][0]), int(keypoints[count][0][0][1]))
            left_wrist = (int(keypoints[count][0][9][0]), int(keypoints[count][0][9][1]))
            left_ankle = (int(keypoints[count][0][15][0]), int(keypoints[count][0][15][1]))
            cv2.circle(img, head, 5, (0, 0, 255), -1)
            cv2.circle(img, left_wrist, 5, (0, 0, 255), -1)
            cv2.circle(img, left_ankle, 5, (0, 0, 255), -1)
            ball = (int(balls[count][0]), int(balls[count][1]))
            cv2.circle(img, ball, 5, (0, 0, 255), -1)
            cv2.putText(img, 'frame:' + str(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'y:' + str(head[1]), head, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'y:' + str(left_wrist[1]), left_wrist, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'y:' + str(left_ankle[1]), left_ankle, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow('frame', img)
            cv2.waitKey(waitkey)
            count += 1
        else:
            break
    cap.release()
    return


def find_first_raise_hand(keypoints, k=10):
    """
    找到第一次举手的关键点索引。

    参数:
    keypoints (np.ndarray): 关键点数组。
    k (int): 连续举手帧数阈值。

    返回:
    int: 第一次举手的关键点索引，如果没有找到则返回-1。
    """
    count = 0
    for i, kpt in enumerate(keypoints):
        head_y = kpt[0][0][1]
        left_wrist_y = kpt[0][9][1]
        if left_wrist_y < head_y:
            if count == k:
                return i - k
            count += 1
        else:
            count = 0
    return -1


def read_npy(npy_path: str) -> np.ndarray:
    """
    读取npy文件。

    参数:
    npy_path (str): npy文件路径。

    返回:
    np.ndarray: 读取的数组数据，如果文件不存在则返回None。
    """
    if os.path.isfile(npy_path):
        return np.load(npy_path)


def find_non_zero(array) -> np.ndarray:
    """
    找到数组中非零元素的索引。

    参数:
    array (np.ndarray): 输入数组。

    返回:
    np.ndarray: 非零元素的索引数组。
    """
    return np.where(~np.all(array == 0, axis=1))[0]


def find_outliers_lof(points: np.ndarray, contamination=0.05) -> np.ndarray:
    """
    使用局部离群因子算法找到离群点。

    参数:
    points (np.ndarray): 输入点数组。
    contamination (float): 离群点比例，默认为0.05。

    返回:
    np.ndarray: 离群点的索引数组。
    """
    points = points.reshape(points.shape[0], -1)
    lof = LocalOutlierFactor(contamination=contamination)
    outliers_idx = np.where(lof.fit_predict(points) == -1)
    return outliers_idx[0]


def find_outlier(array: np.ndarray) -> np.ndarray:
    """
    找到并处理数组中的离群点，将离群点置零。

    参数:
    array (np.ndarray): 输入数组。

    返回:
    np.ndarray: 处理后的数组。
    """
    non_zero_idx = find_non_zero(array)
    array_without_zeros = array[non_zero_idx]
    outliers_idx = find_outliers_lof(array_without_zeros)
    print([non_zero_idx[outliers_idx]])
    array[non_zero_idx[outliers_idx]] = np.array([0, 0])
    return array


def interpolate(array: np.ndarray) -> np.ndarray:
    """
    对数组中的零值进行插值处理。

    参数:
    array (np.ndarray): 输入数组。

    返回:
    np.ndarray: 插值处理后的数组。
    """
    # 处理离群点
    array = find_outlier(array)
    # 插值处理零值
    non_zero_idx = find_non_zero(array)
    zero_idx = np.setdiff1d(np.arange(array.shape[0]), non_zero_idx)
    x_interpolate = np.interp(zero_idx, non_zero_idx, array[non_zero_idx, 0])
    y_interpolate = np.interp(zero_idx, non_zero_idx, array[non_zero_idx, 1])
    for i, idx in enumerate(zero_idx):
        array[idx, 0] = int(x_interpolate[i])
        array[idx, 1] = int(y_interpolate[i])
    return array


def draw_ball(img, ball_pos, color=(0, 0, 255)):
    """
    在图像上绘制球的位置。

    参数:
    img (np.ndarray): 输入图像。
    ball_pos (tuple): 球的位置坐标。
    color (tuple): 绘制颜色，默认为红色。

    返回:
    np.ndarray: 绘制后的图像。
    """
    res = deepcopy(img)
    cv2.circle(res, (int(ball_pos[0]), int(ball_pos[1])), 5, color, -1)
    return res


def cal_ball_y_to_court(balls, lines, frame_count):
    """
    计算球到球场的高度投影。

    参数:
    balls (np.ndarray): 球的位置数组。
    lines (np.ndarray): 球场线条数组。
    frame_count (int): 帧数。

    返回:
    float: 球到球场的高度投影。
    """
    ball_y = balls[frame_count][1]
    x1, y1, x2, y2 = lines[frame_count][4], lines[frame_count][5], lines[frame_count][6], lines[frame_count][7]
    project_y = y1 + (y2 - y1) * (ball_y - x1) / (x2 - x1)
    return project_y - ball_y


def cal_vertical_cross_point(point, line):
    """
    计算垂直交叉点。

    参数:
    point (tuple): 点坐标。
    line (np.ndarray): 线条坐标。

    返回:
    np.ndarray: 垂直交叉点坐标。
    """
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    if x1 != x2:
        y0 = (point[0] - x1) / (x2 - x1) * (y2 - y1) + y1
        return np.array([point[0], y0]).astype(np.int64)


def cal_high_y_ratio(balls, frames, keypoints, indices, player_h):
    """
    计算球的垂直高度比例。

    参数:
    balls (np.ndarray): 球的位置数组。
    frames (list): 视频帧列表。
    keypoints (np.ndarray): 关键点数组。
    indices (int): 索引。
    player_h (float): 运动员身高。

    返回:
    float: 球的垂直高度比例。
    """
    radius = 2
    if indices > radius:
        start = indices
    else:
        start = radius
    end = len(balls) - 1 - radius
    for i in range(start, end):
        balls_window = balls[i - radius: i + radius + 1]
        if balls[i][1] == np.min(balls_window[:, 1]):
            head_y = keypoints[i][0][0][1]
            left_hip_y = keypoints[i][0][11][1]
            return player_h / abs(head_y - left_hip_y)
    return 0


def project_to_court(balls, lines):
    """
    将球的位置投影到球场。

    参数:
    balls (np.ndarray): 球的位置数组。
    lines (np.ndarray): 球场线条数组。

    返回:
    np.ndarray: 投影后的球的位置数组。
    """
    project_balls = copy.deepcopy(balls)
    if balls[0][0] > balls[-1][0]:
        project_balls = np.flip(project_balls, axis=0)
        start = balls[-1]
        end = cal_vertical_cross_point(balls[0], lines[0, 4: 8])
    else:
        start = cal_vertical_cross_point(balls[0], lines[0, 4: 8])
        end = balls[-1]
    project_balls[:, 1] = np.interp(project_balls[:, 0], [start[0], end[0]], [start[1], end[1]])
    project_balls = project_balls.astype(np.int64)
    if balls[0][0] > balls[-1][0]:
        project_balls = np.flip(project_balls, axis=0)
    return project_balls


def cal_ball_acceleration(balls):
    """
    计算球的加速度。

    参数:
    balls -- 包含球的位置的列表或数组

    返回:
    一个包含每帧间球加速度的数组。
    """
    velocity_vector = np.zeros((len(balls) - 1, 2))
    for i in range(len(balls) - 1):
        if np.all(balls[i + 1] == 0):
            balls[i + 1] = balls[i]
        velocity_vector[i] = balls[i + 1] - balls[i]

    velocity_diff = np.zeros(len(velocity_vector) - 1)
    for j in range(len(velocity_vector) - 1):
        velocity_diff[j] = np.linalg.norm(velocity_vector[j + 1] - velocity_vector[j])
    # plt.plot(velocity_diff[indices - 5: indices + 100])
    # plt.show()
    return velocity_diff


def draw_ball_trajectory(video_path: str):
    # 读取视频中的所有帧
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    video_capture.release()

    # 读取网球、骨骼关键点和场地线数据
    file_name, _ = os.path.splitext(os.path.basename(video_path))
    balls = read_npy(f'./output/ball/{file_name}.npy')
    keypoints = read_npy(f'./output/keypoints/{file_name}.npy')
    lines = read_npy(f'./output/line/{file_name}.npy')
    draw_img(video_path, keypoints, balls, 0)

    # 找到第一次举手的帧
    first_raise_hand = find_first_raise_hand(keypoints)

    # 计算实际高度与像素点高度的比例
    player_h = 1.88 / 214 * 93  # 1.88m / 214px * 93px, 运动员身高和运动员躯干
    h_y_ratio = cal_high_y_ratio(balls, frames, keypoints, first_raise_hand, player_h)

    # 插值处理网球轨迹
    balls = interpolate(balls)

    # 若加速度突变且大于阈值，则认为第一个突变帧是发球帧，第二个突变帧是落地帧
    ball_a = cal_ball_acceleration(balls)
    acceleration_threshold = 15.0
    start, end = 0, 0
    for i in range(first_raise_hand, len(balls)):
        if ball_a[i] > np.mean(ball_a[i - 3: i]) and ball_a[i] > acceleration_threshold:
            if start == 0:
                start = i
            elif i > start + 5:
                end = i
                break
    # 计算加速度的值需要用三帧位置，而截取应取加速度突变的前一帧，因此start和end需要加2减1
    start += 1
    end += 1
    serve_balls = balls[start: end]
    serve_lines = lines[start: end]

    # 将球位置投影到地面
    project_balls = project_to_court(serve_balls, serve_lines)

    max_h = cal_ball_y_to_court(balls, lines, start) * h_y_ratio
    project_balls_z = np.linspace(max_h, 0.0, num=len(project_balls))
    print("start: ", start)
    print("end: ", end)
    print("x, y: ", project_balls)
    print("z: ", project_balls_z)
    frame_count = 0

    # heatmap = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # for frame in frames:
    #     if frame_count == len(balls[180:199]):
    #         break
    #     c = int(frame_count * 255 / len(balls[180:199]))
    #     heatmap = draw_ball(heatmap, balls[180:199][frame_count], (0, 0, 255 - c))
    #     frame_count += 1

    # cv2.imshow("heatmap", heatmap)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()


draw_ball_trajectory('videos/001.mp4')
