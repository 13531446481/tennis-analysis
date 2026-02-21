import os
from pathlib import Path
import gradio as gr
import cv2
import numpy as np

from rtmlib.visualization.skeleton import coco17
red = (0, 0, 255)  # OpenCV 的 BGR 红色
from tennis_tracking.Models.tracknet import trackNet
from tennis_tracking.court_detector import CourtDetector

from draw import draw_mmpose
from pose_estimation_rtmlib import PosePredictModel

# 设置相对路径的临时目录
current_dir = Path(__file__).parent.resolve()  # 获取当前脚本所在目录
gradio_temp_dir = current_dir / "tmp" / "gradio_tmp"

# 确保目录存在
gradio_temp_dir.mkdir(parents=True, exist_ok=True)

# 设置环境变量（必须在导入Gradio组件之前）
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

openpose_skeleton = False  # True for openpose-style, False for mmpose-style
model = PosePredictModel(to_openpose=openpose_skeleton,
                         mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                         backend=backend, device=device)

n_classes = 256
save_weights_path = 'tennis_tracking/WeightsTracknet/model.1'
width, height = 640, 360
modelFN = trackNet
tracknet_model = modelFN(n_classes, input_height=height, input_width=width)
tracknet_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
tracknet_model.load_weights(save_weights_path)


def run_predict_model(input_video, input_pose):
    output_video = input_video
    output_pose = input_pose
    return output_video, output_pose


with gr.Blocks() as demo:
    preview_tab_state = gr.State("ground truth")
    points = gr.State([])
    preview_output_state = gr.State({"input_len": 10, "output_len": 5, "input_step": 1, "output_step": 1})
    """
    Page design
    """
    gr.Markdown("# Pose prediction demo")

    with gr.Tab("Video Detect"):
        with gr.Row():
            with gr.Column():
                detect_upload_video = gr.Video(label="Video Input")
                with gr.Row():
                    detect_ball_button = gr.Button("detect ball")
                    detect_line_button = gr.Button("detect line")
                    detect_pose_button = gr.Button("detect pose")
                detect_image = gr.Image(label="mark image", interactive=True)
                with gr.Row():
                    output = gr.Textbox(label="output")
                    add_point = gr.Button("add point")
                    points_show = gr.Textbox(label="points")
                # detect_frame_slider = gr.Slider()
                # detect_run_button = gr.Button("Run")

            with gr.Column():
                detect_output_button = gr.Button("Output")
                detect_video_output = gr.PlayableVideo(label="Video Output")
                with gr.Row():
                    detect_ball_file = gr.File(label="Ball")
                    detect_line_file = gr.File(label="Line")
                    detect_pose_file = gr.File(label="Pose")

    with gr.Tab("Preview & Compare"):
        with gr.Row():
            with gr.Column():
                preview_upload_video = gr.Video(label="Video Input")
                with gr.Row():
                    preview_gt_ball_file = gr.File(label="gt ball", visible=False)
                    preview_gt_line_file = gr.File(label="gt line")
                    preview_gt_pose_file = gr.File(label="gt pose")
                with gr.Row():
                    preview_pred_pose_file = gr.File(label="pred pose")

            with gr.Column():
                with gr.Row():
                    preview_tab_gt = gr.Tab("ground truth")
                    preview_tab_pred = gr.Tab("predict")
                    preview_tab_compare = gr.Tab("compare")
                    preview_predict_frame = gr.Number(value=5, label="total output frame")
                with gr.Row():
                    preview_image = gr.Image(label="preview image")
                with gr.Row():
                    preview_input_slider = gr.Slider(step=1, label="input frame", interactive=True)
                    preview_predict_slider = gr.Slider(minimum=0, maximum=5, step=1, label="predict frame",
                                                       interactive=True)
                    preview_frame_num = gr.Number(label="predict ms")
                with gr.Row():
                    test_tab_text = gr.Textbox(preview_tab_state.value)
                with gr.Row():
                    compare_pose_gt = gr.Image(label="pose predict gt", visible=False)
                with gr.Row():
                    compare_pose_pred = gr.Image(label="pose predict pred", visible=False)

    """
    event handle function
    """


    def draw_first_frame(input_video):
        video_capture = cv2.VideoCapture(input_video)
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_capture.release()
            return frame
        video_capture.release()
        return None


    def get_click_coordinates(frame, point, evt: gr.SelectData):
        point.append(evt.index)
        cv2.circle(frame, (evt.index[0], evt.index[1]), 10, (255, 0, 0), -1)
        return frame, point, point, evt.index


    def get_tab_select(evt: gr.SelectData):
        return evt.value, evt.value


    def frame_count_update(input_video):
        cap = cv2.VideoCapture(input_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return gr.update(maximum=frame_count - 1)


    def draw_preview_image(video, gt_ball, gt_line, gt_pose, pred_pose, input_frame, predict_frame, state):
        if pred_pose:
            step = pred_pose.split('_').index('step')
            output_step = int(pred_pose.split('.')[-2].split('_')[step + 2])
        else:
            output_step = 1
        current_frame = input_frame + predict_frame * output_step
        if not video:
            return None, current_frame
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, image = cap.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = rgb_image.copy()
        cap.release()

        if state == 'ground truth' or state == 'compare':
            if gt_ball:
                pass
            if gt_line:
                gt_line_list = np.load(gt_line).tolist()
                lines = gt_line_list[current_frame]
                for i in range(0, len(lines), 4):
                    x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
                    cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            if gt_pose:
                gt_keypoints = np.load(gt_pose)
                gt_frame_keypoints = gt_keypoints[current_frame]
                for kpt in gt_frame_keypoints:
                    output_image = draw_mmpose(output_image, kpt, coco17['keypoint_info'], coco17['skeleton_info'])

        if state == 'predict' or state == 'compare':
            if pred_pose and input_frame >= 15 and predict_frame >= 1:
                pred_keypoints = np.load(pred_pose)
                pred_frame_keypoints = pred_keypoints[input_frame - 15][predict_frame - 1]
                pred_frame_keypoints = pred_frame_keypoints.reshape(17, 2)
                output_image = draw_mmpose(output_image, pred_frame_keypoints, red['keypoint_info'],
                                           red['skeleton_info'])
        return output_image, int(predict_frame * (100 / 3) * output_step)


    def ball_detect(input_video):
        video_capture = cv2.VideoCapture(input_video)
        output_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 1
        ball_list = []
        while True:
            ret, frame = video_capture.read()
            if ret:
                frame = cv2.resize(frame, (width, height)).astype(np.float32)
                tracknet_input = np.rollaxis(frame, 2, 0)
                tracknet_output = tracknet_model.predict(np.array([tracknet_input]))[0].reshape(
                    (height, width, n_classes)).argmax(axis=2).astype(np.uint8)
                heatmap = cv2.resize(tracknet_output, (output_width, output_height))
                ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2,
                                           minRadius=2,
                                           maxRadius=7)
            else:
                break
            frame_count += 1
            if circles is not None:
                if len(circles) == 1:
                    x = int(circles[0][0][0])
                    y = int(circles[0][0][1])
                    ball_list.append([x, y])
                else:
                    ball_list.append([0, 0])
            else:
                ball_list.append([0, 0])
        video_capture.release()
        return ball_list


    @detect_ball_button.click(
        inputs=[detect_upload_video],
        outputs=[detect_ball_file]
    )
    def detect_ball(input_video):
        balls = ball_detect(input_video)
        balls = np.array(balls)
        ball_path = "./output/ball"
        file_name, _ = os.path.splitext(os.path.basename(input_video))
        ball_file = os.path.join(ball_path, f'{file_name}.npy')
        if not os.path.exists(ball_path):
            os.makedirs(ball_path)
        np.save(ball_file, balls)
        return ball_file


    def court_detect(input_video):
        court_detector = CourtDetector()
        video_capture = cv2.VideoCapture(input_video)
        frame_count = 1
        line_list = []
        while True:
            ret, frame = video_capture.read()
            if ret:
                if frame_count == 1:
                    lines = court_detector.detect(frame)
                else:
                    lines = court_detector.track_court(frame)
            else:
                break
            frame_count += 1
            line_list.append(lines)
        video_capture.release()
        return line_list


    @detect_line_button.click(
        inputs=[detect_upload_video],
        outputs=[detect_line_file]
    )
    def detect_line(input_video):
        lines = court_detect(input_video)
        np.array(lines)
        line_path = "./output/line"
        file_name, _ = os.path.splitext(os.path.basename(input_video))
        line_file = os.path.join(line_path, f'{file_name}.npy')
        if not os.path.exists(line_path):
            os.makedirs(line_path)
        np.save(line_file, lines)
        return line_file


    @detect_output_button.click(
        inputs=[detect_upload_video],
        outputs=[detect_video_output]
    )
    def video_output(input_video):
        ball_list = []
        line_list = []
        keypoints = None
        ball_path = "./output/ball"
        line_path = "./output/line"
        keypoints_path = "./output/keypoints"
        file_name, _ = os.path.splitext(os.path.basename(input_video))
        ball_file = os.path.join(ball_path, f'{file_name}.npy')
        line_file = os.path.join(line_path, f'{file_name}.npy')
        keypoints_file = os.path.join(keypoints_path, f'{file_name}.npy')
        if os.path.isfile(ball_file):
            ball_list = np.load(ball_file).tolist()
        if os.path.isfile(line_file):
            line_list = np.load(line_file).tolist()
        if os.path.isfile(keypoints_file):
            keypoints = np.load(keypoints_file)

        video_capture = cv2.VideoCapture(input_video)

        output_video_path = "./output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if ret:
                if ball_list:
                    ball = ball_list[frame_count]
                    cv2.circle(frame, (int(ball[0]), int(ball[1])), 5, (0, 0, 255), -1)
                if line_list:
                    lines = line_list[frame_count]
                    for i in range(0, len(lines), 4):
                        x1, y1, x2, y2 = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                if keypoints is not None:
                    keypoint = keypoints[frame_count]
                    frame = draw_mmpose(frame, keypoint[0], coco17['keypoint_info'], coco17['skeleton_info'])
                    frame = draw_mmpose(frame, keypoint[1], coco17['keypoint_info'], coco17['skeleton_info'])
                video_writer.write(frame)
                frame_count += 1
            else:
                break

        video_capture.release()
        video_writer.release()
        return output_video_path


    @detect_pose_button.click(
        inputs=[detect_upload_video, points],
        outputs=[detect_pose_file]
    )
    def detect_pose(input_video, point, output_path=None):
        video_capture = cv2.VideoCapture(input_video)
        keypoints_data = []
        while True:
            ret, frame = video_capture.read()
            if ret:
                keypoints, point = model(frame, point)
                keypoints_data.append(keypoints)
            else:
                break
        video_capture.release()
        keypoints_data = np.stack(keypoints_data, axis=0)
        print("keypoints shape : ", keypoints_data.shape)

        if output_path is None:
            output_path = "./output/keypoints"
        file_name, _ = os.path.splitext(os.path.basename(input_video))
        keypoints_file = os.path.join(output_path, f'{file_name}.npy')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(keypoints_file, keypoints_data)

        return keypoints_file


    """
    event function
    """
    # run_model.click(
    #     fn=run_predict_model,
    #     inputs=[video_input, gt_pose],
    #     outputs=[video_output, pred_pose]
    # )
    #
    # play_video_update.click(
    #     fn=frame_count_update,
    #     inputs=video_output,
    #     outputs=slid
    # )
    #
    # slid.change(
    #     fn=play_video,
    #     inputs=[video_output, pred_pose, slid],
    #     outputs=[video_frame, gt_frame, pred_frame, comp_frame]
    # )
    #
    detect_image.select(
        fn=get_click_coordinates,
        inputs=[detect_image, points],
        outputs=[detect_image, points, points_show, output]
    )

    detect_upload_video.upload(
        fn=draw_first_frame,
        inputs=[detect_upload_video],
        outputs=[detect_image]
    )


    @preview_pred_pose_file.upload(
        inputs=[preview_pred_pose_file, preview_output_state],
        outputs=[preview_output_state]
    )
    def update_output_state(file_path, output_state):
        len_idx = file_path.split('_').index('len')
        input_len = int(file_path.split('_')[len_idx + 1])
        output_len = int(file_path.split('_')[len_idx + 2])
        step = file_path.split('_').index('step')
        input_step = int(file_path.split('_')[step + 1])
        output_step = int(file_path.split('.')[-2].split('_')[step + 2])

        output_state['input_len'] = input_len
        output_state['output_len'] = output_len
        output_state['input_step'] = input_step
        output_state['output_step'] = output_step
        return output_state


    @preview_upload_video.change(
        inputs=[preview_upload_video, preview_output_state],
        outputs=[preview_input_slider, preview_predict_frame]
    )
    def update_output_message(video, output_state):
        if video:
            video_capture = cv2.VideoCapture(video)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            input_len = output_state['input_len']
            output_len = output_state['output_len']
            input_step = output_state['input_step']
            output_step = output_state['output_step']
            input_seq = (input_len - 1) * input_step + 1
            output_seq = output_len * output_step
            seq_len = input_seq + output_seq

            input_frame = frame_count - seq_len
            return gr.update(maximum=input_frame), gr.update(value=output_len)


    preview_tabs = [preview_tab_gt, preview_tab_pred, preview_tab_compare]

    for tab in preview_tabs:
        tab.select(
            fn=get_tab_select,
            outputs=[preview_tab_state, test_tab_text]
        )
    preview_predict_frame.change(
        lambda num: gr.update(maximum=num),
        inputs=preview_predict_frame,
        outputs=preview_predict_slider
    )

    draw_preview_inputs = [
        preview_upload_video,
        preview_gt_ball_file,
        preview_gt_line_file,
        preview_gt_pose_file,
        preview_pred_pose_file,
        preview_input_slider,
        preview_predict_slider,
        preview_tab_state
    ]

    preview_input_slider.change(
        fn=draw_preview_image,
        inputs=draw_preview_inputs,
        outputs=[preview_image, preview_frame_num]
    )

    preview_predict_slider.change(
        fn=draw_preview_image,
        inputs=draw_preview_inputs,
        outputs=[preview_image, preview_frame_num]
    )


    @preview_upload_video.upload(
        inputs=[preview_upload_video],
        outputs=[preview_input_slider]
    )
    def update_slider(input_video):
        video_capture = cv2.VideoCapture(input_video)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()
        return gr.Slider(minimum=0, maximum=frame_count, step=1, value=0)

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    debug=True,
    show_error=True,
    prevent_thread_lock=True,
)
