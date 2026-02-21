import os
import re
from copy import deepcopy

import cv2
import numpy as np
import easyocr

from tennis_tracking.Models.tracknet import trackNet
from tennis_tracking.court_detector import CourtDetector


class TennisDetector:
    """
    Detecting and tracking tennis ball and court lines in tennis videos.
    """

    def __init__(
            self,
            tracknet_weights: str,
            n_classes: int = 256,
            height: int = 360,
            width: int = 640
    ):
        self.tracknet_weights = tracknet_weights
        self.n_classes = n_classes
        self.height = height
        self.width = width

        if not os.path.isfile(self.tracknet_weights):
            raise FileNotFoundError(f"Weights file not found at {self.tracknet_weights}")

        self._tracknet_model = trackNet(self.n_classes, input_height=self.height, input_width=self.width)
        self._tracknet_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        try:
            self._tracknet_model.load_weights(self.tracknet_weights)
        except Exception as e:
            raise IOError(f"Error loading weights from {self.tracknet_weights}: {str(e)}")

        self._court_detector = CourtDetector()

    def ball_detect(self, input_video: str, lines: list[np.ndarray]) -> list[list[int]]:
        """
        Tennis ball detection using tracknet.

        Args:
            lines: line.
            input_video: Input video path.

        Returns:
            list of detected ball coordinates, [[x1,y1], [x2,y2], ...].
        """
        if not os.path.isfile(input_video):
            raise FileNotFoundError(f"Video file not found at {input_video}")

        video_capture = cv2.VideoCapture(input_video)
        output_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        ball_list = []
        print("processing ball detect")
        while True:
            ret, frame = video_capture.read()
            frame_view = deepcopy(frame)
            if ret:
                x1, y1 = lines[frame_count][0], lines[frame_count][1]
                x2, y2 = lines[frame_count][2], lines[frame_count][3]
                dx = x2 - x1
                dy = y2 - y1
                # 计算延长后的端点
                pt3 = (int(x1 - 0.05 * dx), int(y1 - 0.05 * dy))
                pt4 = (int(x2 + 0.05 * dx), int(y2 + 0.05 * dy))
                # 有时会将底线识别为网球，将底线遮住再识别
                cv2.line(frame, pt3, pt4, (0, 0, 0), 6)
                # cv2.imshow("frame", frame)
                frame = cv2.resize(frame, (self.width, self.height)).astype(np.float32)
                tracknet_input = np.rollaxis(frame, 2, 0)
                tracknet_output = self._tracknet_model.predict(np.array([tracknet_input]))[0].reshape(
                    (self.height, self.width, self.n_classes)).argmax(axis=2).astype(np.uint8)
                heatmap = cv2.resize(tracknet_output, (output_width, output_height))
                arr = np.zeros_like(heatmap)
                heatmap_bgr = cv2.merge((arr, arr, heatmap))
                mask = (heatmap_bgr != [0, 0, 0]).any(axis=2)
                frame_view[mask] = heatmap_bgr[mask]
                # cv2.imshow("heatmap", frame_view)
                cv2.waitKey(1)
                ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2,
                                           minRadius=2, maxRadius=7)
            else:
                break
            frame_count += 1
            if circles is not None:
                if len(circles) == 1:
                    x = int(circles[0][0][0])
                    y = int(circles[0][0][1])
                    ball_list.append([x, y])
                    # cv2.circle(frame_view, (x, y), 5, (0, 0, 255), -1)
                    # cv2.imshow("frame", frame_view)
                    # cv2.waitKey(1)
                else:
                    ball_list.append([0, 0])
            else:
                ball_list.append([0, 0])
        video_capture.release()
        return ball_list

    def court_detect(self, input_video: str) -> list[np.ndarray]:
        """
        Tennis court detection.

        Args:
            input_video: Input video path.

        Returns:
             List of detected court coordinates, each ndarray [x1, y1, x2, y2, ...] contains 20 points of one frame.
        """
        if not os.path.isfile(input_video):
            raise FileNotFoundError(f"Video file not found at {input_video}")

        self._court_detector = CourtDetector()
        video_capture = cv2.VideoCapture(input_video)
        frame_count = 0
        line_list = []
        print("processing court detect")
        while True:
            ret, frame = video_capture.read()
            if ret:
                if frame_count == 0:
                    lines = self._court_detector.detect(frame)
                else:
                    lines = self._court_detector.track_court(frame)
            else:
                break
            line_list.append(lines)
            frame_count += 1
        video_capture.release()
        return line_list

    def speed_detect(self, input_video: str, input_court: str) -> int:
        """
        Tennis ball speed detection.

        Args:
            input_video: Input video path.
            input_court: Input court detect file.

        Returns:
            Tennis ball speed.
        """
        if not os.path.isfile(input_video):
            raise FileNotFoundError(f"Video file not found at {input_video}")

        if not os.path.isfile(input_court):
            raise FileNotFoundError(f"Court detect file not found at {input_court}")

        reader = easyocr.Reader(['en'])

        lines = np.load(input_court)
        video_capture = cv2.VideoCapture(input_video)
        count = 0
        output_video_path = "./"
        output_video_path = os.path.join(output_video_path, os.path.basename(input_video))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = video_capture.read()
            if ret:
                x1 = round(lines[count][28])
                y1 = round(lines[count][29])
                x2 = round(lines[count][2])
                y2 = round(lines[count][3])
                x3 = round((1.8 * (x2 - x1) + x1))
                y3 = round((1.8 * (y2 - y1) + y1))
                cv2.line(frame, (x1, y1), (x3, y3), (0, 0, 255), 5)
                pt1 = (x3 - 60, y3 - 120)
                pt2 = (x3 + 60, y3 - 40)
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                cropped_frame = frame[y3 - 120:y3 - 40, x3 - 60:x3 + 60]
                cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = reader.readtext(cropped_frame_rgb)
                number_pattern = r'^[0-9\-+.(),]+$'  # 包含数字和常见符号
                cleaned = '0'
                for (bbox, text, confidence) in results:
                    if re.match(number_pattern, text) and confidence >= 0.2:
                        # 后处理：替换易混淆字符
                        cleaned = text.translate(str.maketrans({
                            'O': '0',  # 字母O转数字0
                            'o': '0',
                            'I': '1',  # 字母I转数字1
                            'l': '1',
                            'S': '5',  # 字母S转数字5
                            's': '5',
                            'B': '8'  # 字母B转数字8
                        })).strip()
                        print(cleaned)
                cv2.putText(frame, cleaned, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                video_writer.write(frame)
                count += 1
            else:
                break
        video_capture.release()
        video_writer.release()
        return 0


if __name__ == "__main__":
    def detect():
        detector = TennisDetector(tracknet_weights='tennis_tracking/WeightsTracknet/model.1')
        path_list = ['./videos/001.mp4']
        for video_path in path_list:
            print(video_path)
            lines = detector.court_detect(video_path)
            balls = detector.ball_detect(video_path, lines)
            # print(len(balls), type(balls), type(balls[0]))
            # print(len(lines), type(lines), type(lines[0]))
            # np.array(lines)
            ball_path = "./output/ball"
            line_path = "./output/line"
            file_name, _ = os.path.splitext(os.path.basename(video_path))
            ball_file = os.path.join(ball_path, f'{file_name}.npy')
            line_file = os.path.join(line_path, f'{file_name}.npy')
            balls = np.array(balls)
            lines = np.array(lines)
            print(balls.shape)
            print(lines.shape)

            np.save(ball_file, balls)
            np.save(line_file, lines)

    detect()
    exit(0)

    def read():
        path_list = [
            './videos/001.mp4', './videos/002.mp4', './videos/003.mp4', './videos/004.mp4', './videos/005.mp4',
            './videos/006.mp4', './videos/007.mp4', './videos/008.mp4', './videos/009.mp4', './videos/010.mp4',
            './videos/011.mp4', './videos/012.mp4', './videos/013.mp4', './videos/014.mp4', './videos/015.mp4',
            './videos/016.mp4', './videos/017.mp4', './videos/018.mp4', './videos/019.mp4', './videos/020.mp4',
            './videos/021.mp4', './videos/022.mp4', './videos/023.mp4', './videos/024.mp4', './videos/025.mp4',
            './videos/026.mp4', './videos/027.mp4', './videos/028.mp4', './videos/029.mp4', './videos/030.mp4',
        ]
        line_list = [
            './output/line/001.npy', './output/line/002.npy', './output/line/003.npy', './output/line/004.npy',
            './output/line/005.npy', './output/line/006.npy', './output/line/007.npy', './output/line/008.npy',
            './output/line/009.npy', './output/line/010.npy', './output/line/011.npy', './output/line/012.npy',
            './output/line/013.npy', './output/line/014.npy', './output/line/015.npy', './output/line/016.npy',
            './output/line/017.npy', './output/line/018.npy', './output/line/019.npy', './output/line/020.npy',
            './output/line/021.npy', './output/line/022.npy', './output/line/023.npy', './output/line/024.npy',
            './output/line/025.npy', './output/line/026.npy', './output/line/027.npy', './output/line/028.npy',
            './output/line/029.npy', './output/line/030.npy'
        ]
        for video_path, line_path in zip(path_list, line_list):
            lines = np.load(line_path)
            video_capture = cv2.VideoCapture(video_path)

            # output_video_path = "./output/videos"
            # output_video_path = os.path.join(output_video_path, os.path.basename(video_path))
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            # fps = video_capture.get(cv2.CAP_PROP_FPS)
            # width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            count = 0
            while True:
                ret, frame = video_capture.read()
                if ret:
                    line = lines[count]
                    for i in range(0, len(line), 4):
                        x1, y1, x2, y2 = line[i], line[i + 1], line[i + 2], line[i + 3]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                        # cv2.imshow("court", frame)
                        cv2.waitKey(0)
                    # video_writer.write(frame)
                    count += 1
                else:
                    break

            video_capture.release()
            # video_writer.release()


    # read()

    # detector = TennisDetector(tracknet_weights='tennis_tracking/WeightsTracknet/model.1')
    # video_path = './videos/003.mp4'
    # line_path = "./output/line/003.npy"
    # detector.speed_detect(video_path, line_path)
