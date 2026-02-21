import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统自带黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def process_video(video_path, lower, upper, threshold):
    cap = cv2.VideoCapture(video_path)
    counts = []
    hit_frames = []
    frame_number = 0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for _ in tqdm.tqdm(range(int(frame_count)), desc="processing"):
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            break

        # 创建颜色掩膜并统计符合像素数量
        mask = cv2.inRange(frame, lower, upper)
        count = np.sum(mask > 0)

        # 记录超过阈值的帧
        if threshold[0] < count < threshold[1]:
            counts.append(count)
            hit_frames.append(frame_number)

        frame_number += 1

    cap.release()
    return counts, hit_frames


def visualize_rgb_distribution(frame, lower, upper):
    mask = cv2.inRange(frame, lower, upper)
    matched_pixels = frame[mask > 0]

    if len(matched_pixels) == 0:
        print("没有符合的像素")
        return

    r, g, b = matched_pixels[:, 0], matched_pixels[:, 1], matched_pixels[:, 2]

    # 三维散点图
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(r, g, b, c=matched_pixels / 255, s=10)
    ax.set_xlabel('R'), ax.set_ylabel('G'), ax.set_zlabel('B')
    ax.set_title('RGB三维分布')

    # 二维直方图（R-G平面）
    plt.subplot(122)
    plt.hist2d(r, g, bins=(50, 50), cmap='viridis')
    plt.xlabel('R'), plt.ylabel('G')
    plt.colorbar()
    plt.title('R-G平面分布')

    plt.tight_layout()
    plt.show()


def extract_video_segments(input_path, hit_frames, min_consecutive=100):
    """提取连续帧片段并保存为独立视频

    Args:
        input_path (str): 原始视频路径
        hit_frames (list): 符合条件的帧号列表
        min_consecutive (int): 最小连续帧阈值
    """
    # 排序并分组连续帧
    sorted_frames = sorted(hit_frames)
    segments = []
    current_segment = [sorted_frames[0]]

    for frame in sorted_frames[1:]:
        if frame == current_segment[-1] + 1:
            current_segment.append(frame)
        else:
            if len(current_segment) >= min_consecutive:
                segments.append(current_segment)
            current_segment = [frame]

    # 处理最后一个片段
    if len(current_segment) >= min_consecutive:
        segments.append(current_segment)

    if not segments:
        print("未找到符合条件的连续片段")
        return

    # 读取视频基本信息
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出目录
    output_dir = "./cuts"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有有效片段
    for i, segment in enumerate(segments, 1):
        start_frame = segment[0]
        end_frame = segment[-1]

        # 边界检查
        if start_frame >= total_frames or end_frame >= total_frames:
            print(f"跳过无效片段 {i}：{start_frame}-{end_frame}")
            continue

        # 设置视频写入参数
        output_path = os.path.join(output_dir, f"{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 逐帧读取并写入
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        out.release()
        print(f"已保存片段 {output_path} ({len(segment)}帧)")

    cap.release()


def main():
    # 定义网球颜色范围（示例值，需根据实际情况调整）[BGR]
    lower = np.array([150, 100, 30], dtype=np.uint8)
    upper = np.array([250, 180, 110], dtype=np.uint8)
    threshold = [int(1920 * 1080 * 0.65), int(1920 * 1080 * 0.75)]  # 触发击球检测的像素数量阈值
    print(threshold)

    # 处理视频并检测击球帧
    video_path = './Stefanos Tsitsipas v Novak Djokovic Full Match  Australian Open 2023 Final.mp4'
    # video_path = 'D:/projects/personal-projects/tennis-dataset/澳网2023/Stefanos Tsitsipas v Novak Djokovic Full Match  Australian Open 2023 Final.mp4'
    counts, hit_frames = process_video(video_path, lower, upper, threshold)
    extract_video_segments(video_path, hit_frames)

    # 获取第一个击球帧的像素分布
    # cap = cv2.VideoCapture(video_path)
    # if hit_frames:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frames[0])
    #     ret, sample_frame = cap.read()
    #     if ret:
    #         visualize_rgb_distribution(sample_frame, lower, upper)
    # cap.release()

    # 绘制像素数量变化曲线
    # plt.figure()
    # plt.plot(counts)
    # plt.xlabel('帧号'), plt.ylabel('符合像素数量')
    # plt.axhline(y=threshold, color='r', linestyle='--', label='阈值')
    # plt.title('像素数量随时间变化')
    # plt.legend()
    # plt.show()


main()

# 全局变量用于记录鼠标状态
drawing = False
ix, iy = -1, -1
current_x, current_y = -1, -1
img = None
img_copy = None


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_x, current_y, img, img_copy

    # 鼠标左键按下：开始绘制矩形
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_x, current_y = x, y
        img_copy = img.copy()  # 保存原始图像的副本

    # 鼠标移动：实时更新矩形框
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_x, current_y = x, y
            temp_img = img_copy.copy()
            # 绘制半透明绿色矩形
            cv2.rectangle(temp_img, (ix, iy), (current_x, current_y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_img)

    # 鼠标左键释放：计算选中区域
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_x, current_y = x, y

        # 计算有效区域坐标
        x1 = min(ix, current_x)
        y1 = min(iy, current_y)
        x2 = max(ix, current_x)
        y2 = max(iy, current_y)

        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # 确保区域有效
        if x2 > x1 and y2 > y1:
            # 提取选中区域（注意OpenCV使用BGR格式）
            roi = img[y1:y2, x1:x2]

            # 转换为RGB格式计算
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # 计算各通道极值
            min_vals = np.min(roi_rgb, axis=(0, 1))
            max_vals = np.max(roi_rgb, axis=(0, 1))

            # 格式化输出结果
            print("\n选中区域RGB分析：")
            print(f"  R通道：最小值={min_vals[0]:3d}, 最大值={max_vals[0]:3d}")
            print(f"  G通道：最小值={min_vals[1]:3d}, 最大值={max_vals[1]:3d}")
            print(f"  B通道：最小值={min_vals[2]:3d}, 最大值={max_vals[2]:3d}")

            # 在原始图像上绘制最终矩形
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

# 主程序
# if __name__ == "__main__":
#     # 读取图片（替换为你的图片路径）
#     img_path = "./tennis/imageFiles/001/001_0000.jpg"
#     img = cv2.imread(img_path)
#
#     if img is None:
#         print(f"错误：无法读取图片 {img_path}")
#         exit()
#
#     img_copy = img.copy()
#     cv2.namedWindow('Image')
#     cv2.setMouseCallback('Image', mouse_callback)
#
#     # 显示初始图像
#     cv2.imshow('Image', img)
#     print("操作指南：")
#     print("1. 按住左键拖动鼠标框选区域")
#     print("2. 松开左键自动显示分析结果")
#     print("3. 按ESC键退出程序")
#
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC键退出
#             break
#
#     cv2.destroyAllWindows()
