import os
from typing import Union
import cv2
from tqdm import tqdm
import numpy as np

from rtmlib.rtmlib import Body
from rtmlib.rtmlib.visualization import draw_skeleton


class PosePredictModel(Body):
    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'performance',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super(PosePredictModel, self).__init__(det=det,
                                               det_input_size=det_input_size,
                                               pose=pose,
                                               pose_input_size=pose_input_size,
                                               mode=mode,
                                               to_openpose=to_openpose,
                                               backend=backend,
                                               device=device)

    def __call__(self, image: np.ndarray, hint_points: Union[list, tuple] = None, dis_threshold: Union[int, float] = 1000):
        keypoints, scores = super(PosePredictModel, self).__call__(image)

        if hint_points is None:
            return keypoints, hint_points

        if isinstance(hint_points, tuple):
            assert len(hint_points) == 2, \
                f"hint_points should be (x, y) or [p1, ..., pn], get tuple length {len(hint_points)}"
            hint_points = [hint_points]
        elif isinstance(hint_points, list):
            assert all(len(item) == 2 for item in hint_points), \
                f"hint_points should be (x, y) or [p1, ..., pn], get list item length != 2"
        else:
            raise AssertionError(
                f"hint_points should be tuple, list or None, get type {type(hint_points)}"
            )

        new_keypoints = []
        new_hint_points = []
        for h_point in hint_points:
            distance = []
            h_point = np.array(h_point)
            for kpt in keypoints:
                left_right_foot = kpt[-2:]
                foot_mid = (left_right_foot[0] + left_right_foot[1]) / 2
                distance.append(cv2.norm(foot_mid - h_point))
            if min(distance) < dis_threshold:
                new_keypoints.append(keypoints[np.argmin(distance)])
            new_hint_points = np.array(new_keypoints)[:, -1, :].tolist()
        return new_keypoints, new_hint_points


if __name__ == '__main__':
    device = 'cuda'  # cpu, cuda, mps
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    image_path = './tennis/imageFiles'
    keypoints_path = './tennis/keypoints'

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    points = [[960, 328], [960, 813]]

    model = PosePredictModel(to_openpose=openpose_skeleton,
                             mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                             backend=backend, device=device)

    path_list = os.listdir(image_path)
    path_list.sort()
    for path in path_list:
        keypoints_data = []
        keypoints = []

        image_list = os.listdir(os.path.join(image_path, path))
        image_list.sort()
        for images in tqdm(image_list, desc="Processing images"):
            img = cv2.imread(os.path.join(image_path, path, images))
            keypoints, points = model(img, points)
            print(keypoints)
            print(points)
            keypoints_data.append(keypoints)
        keypoints_data = np.stack(keypoints_data, axis=0)
        print(keypoints_data.size())

# body = Body(to_openpose=openpose_skeleton,
#             mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
#             backend=backend, device=device)
#
# top_mid = [960, 328]
# bottom_mid = [960, 813]
#
# top_distance = []
# bottom_distance = []
#
# path_list = os.listdir(image_path)
# path_list.sort()
# for path in path_list:
#     keypoints_data = []
#
#     image_list = os.listdir(os.path.join(image_path, path))
#     image_list.sort()
#     for images in tqdm(image_list, desc="Processing images"):
#         img = cv2.imread(os.path.join(image_path, path, images))
#         keypoints, scores = body(img)
#         for kpt in keypoints:
#             left_right_foot = kpt[-2:]
#             foot_mid = (left_right_foot[0] + left_right_foot[1]) / 2
#             top_distance.append(cv2.norm(foot_mid - top_mid))
#             bottom_distance.append(cv2.norm(foot_mid - bottom_mid))
#
#         indices = [np.argmin(top_distance), np.argmin(bottom_distance)]
#         new_keypoints = keypoints[indices]
#         new_scores = scores[indices]
#
#         top_mid = new_keypoints[0][-1]
#         bottom_mid = new_keypoints[1][-1]
#         top_distance = []
#         bottom_distance = []
#
#         keypoints_data.append(new_keypoints)
#
#     # keypoints_data: (f,2,17,2) -> (frame, person, joint, xy)
#     keypoints_data = np.stack(keypoints_data, axis=0)
#     np.save(os.path.join(keypoints_path, f'{path}.npy'), keypoints_data)

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

# img_show = draw_skeleton(img, new_keypoints, new_scores, kpt_thr=0.5)


# cv2_imshow(img_show)
