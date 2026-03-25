import argparse
from pathlib import Path

import cv2
import numpy as np

from court_detector import CourtDetector


def load_ball_csv(csv_path: Path):
    import csv

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["Frame"]))
            vis = int(float(row["Visibility"]))
            x = float(row["X"])
            y = float(row["Y"])
            rows.append((frame, vis, x, y))

    tmax = max(r[0] for r in rows)
    xy = np.full((tmax + 1, 2), np.nan, dtype=np.float32)
    vis = np.zeros((tmax + 1,), dtype=np.uint8)
    for frame, v, x, y in rows:
        if v == 1 and x >= 0 and y >= 0:
            xy[frame] = (x, y)
            vis[frame] = 1
    return xy, vis


def homography_from_line20(court20: np.ndarray) -> np.ndarray:
    src = np.array([court20[2], court20[9], court20[8], court20[6]], dtype=np.float32)
    dst = np.array([[0.0, 23.77], [8.23, 23.77], [8.23, 0.0], [0.0, 0.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def img_to_world(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = xy.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def get_hit_xy(players: np.ndarray, frame_idx: int, server_id: int):
    p = players[frame_idx, server_id]
    l = p[15].astype(np.float32)
    r = p[16].astype(np.float32)
    lv = np.all(np.isfinite(l)) and np.linalg.norm(l) > 1e-6
    rv = np.all(np.isfinite(r)) and np.linalg.norm(r) > 1e-6
    if lv and rv:
        return (l + r) * 0.5
    if lv:
        return l
    if rv:
        return r
    raise RuntimeError("invalid ankle points")


def get_bounce_xy(ball_xy: np.ndarray, ball_vis: np.ndarray, frame_idx: int):
    if 0 <= frame_idx < len(ball_vis) and ball_vis[frame_idx] == 1:
        return ball_xy[frame_idx]
    for d in range(1, 8):
        for t in (frame_idx - d, frame_idx + d):
            if 0 <= t < len(ball_vis) and ball_vis[t] == 1:
                return ball_xy[t]
    raise RuntimeError("no valid bounce ball point")


def score_pair(hit_w: np.ndarray, bounce_w: np.ndarray):
    # Prefer both points inside singles court and hit_y > bounce_y.
    def in_penalty(p):
        x, y = float(p[0]), float(p[1])
        px = 0.0
        if x < 0:
            px += abs(x)
        if x > 8.23:
            px += abs(x - 8.23)
        if y < 0:
            px += abs(y)
        if y > 23.77:
            px += abs(y - 23.77)
        return px

    penalty = in_penalty(hit_w) + in_penalty(bounce_w)
    if hit_w[1] <= bounce_w[1]:
        penalty += 20.0
    # Bounce for near-camera serve should usually be in far half.
    if bounce_w[1] > 11.885:
        penalty += (bounce_w[1] - 11.885) * 0.5
    return float(penalty)


def parse_args():
    p = argparse.ArgumentParser(description="Detect best court line for known hit/bounce event")
    p.add_argument("--video_path", required=True)
    p.add_argument("--out_npy", required=True)
    p.add_argument("--players_npy", default="output/pose_keypoints/2_keypoints.npy")
    p.add_argument("--ball_csv", required=True)
    p.add_argument("--hit_frame", type=int, required=True)
    p.add_argument("--bounce_frame", type=int, required=True)
    p.add_argument("--server_id", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video_path)
    out_npy = Path(args.out_npy)
    players_npy = Path(args.players_npy)
    ball_csv = Path(args.ball_csv)

    players = np.load(players_npy)
    ball_xy, ball_vis = load_ball_csv(ball_csv)

    hit_xy = get_hit_xy(players, args.hit_frame, args.server_id)
    bounce_xy = get_bounce_xy(ball_xy, ball_vis, args.bounce_frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    candidates = sorted(set([0, 5, 10, 15, 20, 30, 40, 50, 60, args.hit_frame, args.bounce_frame]))
    candidates = [c for c in candidates if 0 <= c < total]

    detector = CourtDetector(verbose=0)
    best_line = None
    best_frame = None
    best_score = 1e9

    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            line = detector.detect(frame, verbose=0)
            line = np.asarray(line, dtype=np.float32).reshape(-1)
            if line.shape[0] != 40 or not np.isfinite(line).all():
                continue
            H = homography_from_line20(line.reshape(20, 2))
            world = img_to_world(H, np.array([hit_xy, bounce_xy], dtype=np.float32))
            s = score_pair(world[0], world[1])
            if s < best_score:
                best_score = s
                best_line = line
                best_frame = idx
        except Exception:
            continue

    cap.release()

    if best_line is None:
        raise RuntimeError("Failed to detect any valid court line candidate")

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    arr = np.tile(best_line.reshape(1, 40), (max(total, 1), 1)).astype(np.float32)
    np.save(out_npy, arr)

    H = homography_from_line20(best_line.reshape(20, 2))
    world = img_to_world(H, np.array([hit_xy, bounce_xy], dtype=np.float32))

    print(f"[OK] best_frame={best_frame}, score={best_score:.4f}")
    print(f"[OK] saved {out_npy} shape={arr.shape}")
    print(f"[OK] hit_world=({world[0,0]:.4f},{world[0,1]:.4f}) bounce_world=({world[1,0]:.4f},{world[1,1]:.4f})")


if __name__ == "__main__":
    main()
