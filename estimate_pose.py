import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from rtmlib import Body


def dump_pose_from_video(
    video_path: str,
    out_dir: str,
    device: str = "cuda",              # cpu/cuda/mps
    backend: str = "onnxruntime",      # opencv/onnxruntime/openvino
    mode: str = "performance",         # performance/lightweight/balanced
    to_openpose: bool = False,
    max_frames: int = -1,              # -1 means all
):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)

    # model
    body = Body(
        to_openpose=to_openpose,
        mode=mode,
        backend=backend,
        device=device,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # per-frame storage (variable people count -> list)
    all_kpts = []    # each element: (P, J, 2) float32
    all_scores = []  # each element: (P, J) float32

    pbar_total = total if total is not None else 0
    pbar = tqdm(total=pbar_total, desc="Pose estimation (video)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames != -1 and frame_idx > max_frames:
            break

        keypoints, scores = body(frame)  # keypoints: list/ndarray, scores: list/ndarray

        # normalize to numpy arrays with safe empty
        if keypoints is None or len(keypoints) == 0:
            kpt_arr = np.zeros((0, 0, 2), dtype=np.float32)
            scr_arr = np.zeros((0, 0), dtype=np.float32)
        else:
            kpt_arr = np.array(keypoints, dtype=np.float32)  # (P, J, 2)
            scr_arr = np.array(scores, dtype=np.float32)     # (P, J)

        all_kpts.append(kpt_arr)
        all_scores.append(scr_arr)

        pbar.update(1 if total is None else 1)

    pbar.close()
    cap.release()

    # Save as object arrays (variable-length per frame)
    kpts_obj = np.array(all_kpts, dtype=object)
    scores_obj = np.array(all_scores, dtype=object)

    # 1) npz (keypoints + scores + meta)
    npz_path = os.path.join(out_dir, "pose_dump_all_frames.npz")
    np.savez_compressed(
        npz_path,
        keypoints=kpts_obj,
        scores=scores_obj,
        fps=np.array([fps], dtype=np.float32),
        width=np.array([w], dtype=np.int32),
        height=np.array([h], dtype=np.int32),
        num_frames=np.array([len(all_kpts)], dtype=np.int32),
    )

    # 2) npy only keypoints (often enough)
    npy_path = os.path.join(out_dir, "pose_keypoints_only.npy")
    np.save(npy_path, kpts_obj, allow_pickle=True)

    print(f"\nSaved:")
    print(f"  NPZ (kpts+scores+meta): {npz_path}")
    print(f"  NPY (kpts only):        {npy_path}")
    print(f"Frames processed: {len(all_kpts)}")


if __name__ == "__main__":
    # default paths relative to project root
    project_root = Path(__file__).resolve().parent
    video_path = str(project_root / "videos" / "001.mp4")
    out_dir = str(project_root / "output" / "pose_keypoints")

    dump_pose_from_video(
        video_path=video_path,
        out_dir=out_dir,
        device="cuda",
        backend="onnxruntime",
        mode="performance",
        to_openpose=False,
        max_frames=-1,         # -1 跑完整个视频
    )