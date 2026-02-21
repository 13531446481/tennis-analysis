import argparse
import math
import os

import cv2
import numpy as np


def read_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return float(fps), W, H


def load_ball(ball_path: str):
    b = np.load(ball_path)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"ball npy must be (T,2); got {b.shape}")
    mask = (b[:, 0] != 0) & (b[:, 1] != 0) & np.isfinite(b[:, 0]) & np.isfinite(b[:, 1])
    return b, mask


def load_keypoints(kpt_path: str):
    k = np.load(kpt_path)
    k = np.asarray(k)
    if k.ndim < 3 or k.shape[1] == 0:
        raise ValueError(f"keypoints npy unexpected shape {k.shape}")
    return k


def interpolate_2d(points: np.ndarray, mask: np.ndarray):
    T = points.shape[0]
    out = points.copy()
    idx = np.arange(T)
    for d in range(2):
        valid = mask & np.isfinite(points[:, d])
        if valid.sum() < 5:
            raise RuntimeError("Too few valid ball detections to interpolate.")
        out[:, d] = np.interp(idx, idx[valid], points[valid, d])
    return out


def smooth_2d(uv: np.ndarray, ksize: int = 7):
    k = int(ksize)
    if k < 3:
        return uv
    if k % 2 == 0:
        k += 1
    pad = k // 2
    out = uv.copy().astype(np.float64)
    for d in range(2):
        x = out[:, d]
        xp = np.pad(x, (pad, pad), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(xp, k)
        out[:, d] = np.median(win, axis=1)
    return out


def _line_angle_deg(line4: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line4)
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def pick_ground_line_y(lines10: np.ndarray):
    lines = np.asarray(lines10, dtype=np.float32).reshape(-1, 4)
    if lines.shape != (10, 4):
        return None
    angles = np.array([_line_angle_deg(l) for l in lines], dtype=np.float32)
    horiz_score = np.minimum(np.abs(angles), np.abs(np.abs(angles) - 180.0))
    is_h = horiz_score < 15.0
    h_idx = np.where(is_h)[0]
    if h_idx.size == 0:
        return None
    mid_y = 0.5 * (lines[h_idx, 1] + lines[h_idx, 3])
    return float(np.max(mid_y))


def pick_segment_from_accel(
    uv: np.ndarray,
    fps: float,
    min_len_s: float,
    max_len_s: float,
):
    T = uv.shape[0]
    v = (uv[1:] - uv[:-1]) * fps
    a = (v[1:] - v[:-1]) * fps
    acc = np.linalg.norm(a, axis=1)

    if acc.size < 5:
        return 0, T

    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med)) + 1e-6)
    thr = med + 8.0 * 1.4826 * mad
    peaks = np.where(acc > thr)[0]
    if peaks.size == 0:
        spd = np.linalg.norm(v, axis=1)
        peak = int(np.argmax(spd))
    else:
        peak = int(peaks[np.argmax(acc[peaks])])

    start = max(0, peak - 1)
    min_len = int(round(min_len_s * fps))
    max_len = int(round(max_len_s * fps))
    end_min = min(T, start + max(min_len, 2))
    end_max = min(T, start + max_len)

    end = None
    if peaks.size > 1:
        for p in peaks[np.argsort(-acc[peaks])][1:]:
            f = int(p) + 1
            if end_min <= f <= end_max:
                end = f
                break
    if end is None:
        end = end_max
    if end <= start + 1:
        end = min(T, start + 2)
    return start, end


def pick_toss_start_from_keypoints(
    uv: np.ndarray,
    keypoints: np.ndarray,
    min_consecutive: int = 4,
):
    """
    Toss start: first frame where ball is higher (smaller y) than wrist.
    Use both wrists if available (min y).
    """
    T = min(len(uv), keypoints.shape[0])
    if T == 0:
        return None

    # keypoints layout matches mmpose coco17 in this repo
    # keypoints[t, person, 17, 2]
    def get_wrist_y(kpt_frame):
        # prefer person 0, fallback to person 1 if missing
        for person in range(kpt_frame.shape[0]):
            kpt = kpt_frame[person]
            if kpt.shape[0] < 11:
                continue
            lw = kpt[9]
            rw = kpt[10]
            ys = []
            if np.isfinite(lw[1]) and lw[1] > 0:
                ys.append(float(lw[1]))
            if np.isfinite(rw[1]) and rw[1] > 0:
                ys.append(float(rw[1]))
            if ys:
                return min(ys)
        return None

    count = 0
    for i in range(T):
        by = float(uv[i, 1])
        if not np.isfinite(by) or by <= 0:
            count = 0
            continue
        wy = get_wrist_y(keypoints[i])
        if wy is None:
            count = 0
            continue
        if by < wy:
            count += 1
            if count >= min_consecutive:
                return i - min_consecutive + 1
        else:
            count = 0
    return None


def cut_video_segment(video_path: str, out_path: str, start: int, end: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx < end:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        idx += 1

    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser(description="Cut serve->bounce segment using ball + court lines")
    ap.add_argument("--video", required=True)
    ap.add_argument("--ball", required=True, help="Ball detections .npy (T,2)")
    ap.add_argument("--pose", default=None, help="Pose keypoints .npy (optional)")
    ap.add_argument("--line", default=None, help="Court lines .npy (optional)")
    ap.add_argument("--out", default="cuts/serve_segment.mp4")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--min_len_s", type=float, default=0.25)
    ap.add_argument("--max_len_s", type=float, default=1.2)
    ap.add_argument("--y_margin", type=float, default=10.0)
    ap.add_argument("--toss_k", type=int, default=4, help="Consecutive frames ball above wrist")
    args = ap.parse_args()

    fps, _, _ = read_video_meta(args.video)
    ball_raw, mask = load_ball(args.ball)
    uv = interpolate_2d(ball_raw, mask)
    uv = smooth_2d(uv, ksize=args.smooth)

    start, end = pick_segment_from_accel(
        uv,
        fps,
        min_len_s=args.min_len_s,
        max_len_s=args.max_len_s,
    )

    # If keypoints are provided, override start with toss start
    if args.pose and os.path.isfile(args.pose):
        try:
            keypoints = load_keypoints(args.pose)
            toss_start = pick_toss_start_from_keypoints(
                uv,
                keypoints,
                min_consecutive=args.toss_k,
            )
            if toss_start is not None:
                start = int(toss_start)
        except Exception as e:
            print(f"[warn] pose not usable for toss start: {e}")

    # if court lines are available, refine end using ground line proximity
    if args.line and os.path.isfile(args.line):
        lines = np.load(args.line, allow_pickle=True)
        if len(lines) > 0:
            lines0 = np.asarray(lines[0], dtype=np.float32).reshape(-1, 4)
            ground_y = pick_ground_line_y(lines0)
            if ground_y is not None:
                min_end = max(start + int(round(args.min_len_s * fps)), start + 2)
                max_end = min(len(uv), start + int(round(args.max_len_s * fps)))
                for i in range(min_end, max_end):
                    if uv[i, 1] >= (ground_y - args.y_margin):
                        end = i + 1
                        break

    end = min(max(end, start + 2), len(uv))
    cut_video_segment(args.video, args.out, start, end)

    print("=== Serve segment cut ===")
    print(f"frames [{start}, {end}) len={end - start}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
