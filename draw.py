import cv2


def draw_mmpose(img,
                keypoints,
                keypoint_info,
                skeleton_info,
                radius=2,
                line_width=2):
    assert len(keypoints.shape) == 2

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                         kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        link_color = ske_info['color']
        kpt0 = keypoints[pt0]
        kpt1 = keypoints[pt1]

        img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                       (int(kpt1[0]), int(kpt1[1])),
                       link_color,
                       thickness=line_width)

    return img
