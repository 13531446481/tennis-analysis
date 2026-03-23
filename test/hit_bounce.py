import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def moving_average_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(xp, ker, mode="valid")


def robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + 1e-6
    return (x - med) / scale


def load_ball_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Unexpected csv format: {csv_path}")
        needed = {"Frame", "Visibility", "X", "Y"}
        if not needed.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"Unexpected csv format: {csv_path}")
        for row in reader:
            try:
                fr = int(float(row["Frame"]))
                vi = int(float(row["Visibility"]))
                x = float(row["X"])
                y = float(row["Y"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append((fr, vi, x, y))

    if not rows:
        raise RuntimeError(f"Empty csv: {csv_path}")

    frames = np.array([r[0] for r in rows], dtype=np.int32)
    vis = np.array([r[1] for r in rows], dtype=np.int32)
    xs = np.array([r[2] for r in rows], dtype=np.float32)
    ys = np.array([r[3] for r in rows], dtype=np.float32)

    shift = 1 if frames.min() == 1 else 0
    T = int(frames.max() - shift + 1)

    xy = np.full((T, 2), np.nan, dtype=np.float32)
    v = np.zeros((T,), dtype=np.uint8)
    for fr, vi, x, y in zip(frames, vis, xs, ys):
        idx = int(fr - shift)
        if idx < 0 or idx >= T:
            continue
        if vi == 1 and x >= 0 and y >= 0:
            xy[idx] = (x, y)
            v[idx] = 1

    return xy, v


def load_ball_npy(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected npy shape: {arr.shape}")

    xy = arr.astype(np.float32)
    vis = (~((xy[:, 0] == 0) & (xy[:, 1] == 0))).astype(np.uint8)
    xy[vis == 0] = np.nan
    return xy, vis


def interpolate_short_gaps(xy: np.ndarray, vis: np.ndarray, max_gap: int = 6) -> np.ndarray:
    out = xy.copy().astype(np.float32)
    idx = np.arange(len(out))
    valid = idx[vis.astype(bool)]
    if len(valid) < 2:
        return out

    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        pa = out[a]
        pb = out[b]
        for t in range(1, gap + 1):
            r = t / float(gap + 1)
            out[a + t] = pa + r * (pb - pa)

    return out


def smooth_xy(xy: np.ndarray, k: int = 5) -> np.ndarray:
    x = moving_average_1d(xy[:, 0], k)
    y = moving_average_1d(xy[:, 1], k)
    return np.stack([x, y], axis=1)


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def compute_signals(xy: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = len(xy)
    vel = np.zeros((T, 2), dtype=np.float32)
    speed = np.zeros((T,), dtype=np.float32)
    acc = np.zeros((T,), dtype=np.float32)
    dtheta = np.zeros((T,), dtype=np.float32)

    for t in range(1, T):
        if np.any(np.isnan(xy[t])) or np.any(np.isnan(xy[t - 1])):
            continue
        vel[t] = xy[t] - xy[t - 1]
        speed[t] = np.linalg.norm(vel[t])

    for t in range(2, T):
        acc[t] = np.linalg.norm(vel[t] - vel[t - 1])
        dtheta[t] = angle_deg(vel[t - 1], vel[t])

    valid = np.zeros((T,), dtype=bool)
    for t in range(2, T - 1):
        valid[t] = bool(vis[t - 1] and vis[t] and vis[t + 1])

    return speed, acc, dtheta, valid


def load_players(players_path: str) -> np.ndarray:
    players = np.load(players_path)
    if players.ndim != 4 or players.shape[1] < 2 or players.shape[2] < 11 or players.shape[3] != 2:
        raise RuntimeError(f"Unexpected players shape: {players.shape}")
    return players.astype(np.float32)


def estimate_toss_gate_frame(
    xy: np.ndarray,
    vis: np.ndarray,
    players: np.ndarray,
    margin_px: float = 6.0,
    min_consecutive: int = 2,
) -> Optional[int]:
    T = min(len(xy), len(players), len(vis))
    if T <= 0:
        return None

    cond = np.zeros((T,), dtype=bool)
    for t in range(T):
        if not vis[t] or np.any(np.isnan(xy[t])):
            continue
        ball_y = float(xy[t, 1])

        heads = np.array([
            players[t, 0, 0, 1],
            players[t, 1, 0, 1],
        ], dtype=np.float32)
        heads = heads[~np.isnan(heads)]
        if heads.size == 0:
            continue

        # Smaller y means visually higher in image coordinates.
        top_head_y = float(np.min(heads))
        cond[t] = ball_y + margin_px < top_head_y

    run = 0
    for t in range(T):
        if cond[t]:
            run += 1
            if run >= min_consecutive:
                return t - min_consecutive + 1
        else:
            run = 0

    return None


def find_peak_indices(score: np.ndarray, valid: np.ndarray, q: float, min_dist: int) -> List[int]:
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return []
    thr = float(np.quantile(score[idx], q))
    peaks: List[int] = []

    for t in range(2, len(score) - 2):
        if not valid[t]:
            continue
        if score[t] < thr:
            continue
        if score[t] >= score[t - 1] and score[t] >= score[t + 1]:
            if peaks and t - peaks[-1] < min_dist:
                if score[t] > score[peaks[-1]]:
                    peaks[-1] = t
            else:
                peaks.append(t)

    return peaks


def pick_hit_bounce(
    xy: np.ndarray,
    speed: np.ndarray,
    acc: np.ndarray,
    dtheta: np.ndarray,
    valid: np.ndarray,
    fps: float,
    start_frame: int = 0,
) -> Optional[Tuple[int, int]]:
    y = xy[:, 1]
    dy = np.diff(y, prepend=y[0])

    hit_score = 0.50 * robust_z(acc) + 0.30 * robust_z(dtheta) + 0.20 * robust_z(np.maximum(0.0, np.diff(speed, prepend=speed[0])))
    bounce_turn = np.zeros_like(speed, dtype=np.float32)
    for t in range(2, len(y) - 1):
        if np.isnan(y[t - 1]) or np.isnan(y[t]) or np.isnan(y[t + 1]):
            continue
        # In image coordinates, bounce often appears as local max in y.
        if dy[t] > 0 and dy[t + 1] < 0:
            bounce_turn[t] = 1.0

    bounce_score = 0.55 * robust_z(acc) + 0.25 * robust_z(dtheta) + 0.20 * robust_z(bounce_turn)

    if start_frame > 0:
        gate = np.zeros_like(valid, dtype=bool)
        gate[start_frame:] = True
        valid = valid & gate

    hit_peaks = find_peak_indices(hit_score, valid, q=0.88, min_dist=max(3, int(0.08 * fps)))
    bounce_peaks = find_peak_indices(bounce_score, valid, q=0.84, min_dist=max(3, int(0.08 * fps)))

    if not hit_peaks or not bounce_peaks:
        return None

    min_gap = max(3, int(0.08 * fps))
    max_gap = max(10, int(1.20 * fps))
    expected_gap = 0.36 * fps

    best_pair = None
    best_val = -1e9

    for h in hit_peaks:
        for b in bounce_peaks:
            gap = b - h
            if gap < min_gap or gap > max_gap:
                continue
            val = (
                hit_score[h] * 0.9
                + bounce_score[b] * 1.0
                - 0.06 * abs(gap - expected_gap)
                + 0.25 * bounce_turn[b]
            )
            if val > best_val:
                best_val = float(val)
                best_pair = (h, b)

    if best_pair is None:
        return None
    return best_pair


def get_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def resolve_ball_path(project_root: Path, video_id: str, user_ball_path: str) -> str:
    if user_ball_path:
        return user_ball_path

    candidates = [
        project_root / "output" / "tracknetv4" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4_pytorch" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "ball" / f"{video_id}.npy",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(f"No ball file found for id={video_id}")


def save_result_csv(output_csv: str, video_id: str, hit: int, bounce: int, fps: float) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video_id", "hit", "bounce", "hit_sec", "bounce_sec"],
        )
        w.writeheader()
        w.writerow(
            {
                "video_id": video_id,
                "hit": int(hit),
                "bounce": int(bounce),
                "hit_sec": round(float(hit / fps), 3),
                "bounce_sec": round(float(bounce / fps), 3),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect hit and bounce frames from ball trajectory")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--ball", default="", help="Optional ball file path (.csv or .npy)")
    parser.add_argument("--players", default="", help="Optional players_only.npy path")
    parser.add_argument("--out", default="", help="Optional output csv path")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    project_root = Path(__file__).resolve().parents[1]
    video_id = video_path.stem
    ball_path = resolve_ball_path(project_root, video_id, args.ball)

    if ball_path.endswith(".csv"):
        xy, vis = load_ball_csv(ball_path)
    elif ball_path.endswith(".npy"):
        xy, vis = load_ball_npy(ball_path)
    else:
        raise RuntimeError(f"Unsupported ball file: {ball_path}")

    xy_i = interpolate_short_gaps(xy, vis, max_gap=6)
    xy_s = smooth_xy(xy_i, k=5)
    fps = get_fps(str(video_path))

    players_path = args.players or str(project_root / "output" / "pose_keypoints" / "players_only.npy")
    toss_gate = 0
    if os.path.isfile(players_path):
        try:
            players = load_players(players_path)
            gate = estimate_toss_gate_frame(xy_s, vis, players, margin_px=8.0, min_consecutive=2)
            if gate is not None:
                toss_gate = int(gate)
        except Exception as e:
            print(f"[WARN] toss gate disabled due to players file issue: {e}")
    else:
        print("[WARN] players file not found, toss gate disabled")

    speed, acc, dtheta, valid = compute_signals(xy_s, vis)
    pair = pick_hit_bounce(xy_s, speed, acc, dtheta, valid, fps, start_frame=toss_gate)
    if pair is None:
        raise RuntimeError("Failed to detect hit/bounce pair")

    hit, bounce = pair
    if bounce <= hit:
        raise RuntimeError(f"Invalid pair: hit={hit}, bounce={bounce}")

    output_csv = args.out
    if not output_csv:
        output_csv = str(project_root / "output" / "ball" / f"hit_bounce_{video_id}.csv")

    save_result_csv(output_csv, video_id, hit, bounce, fps)

    print(f"video={video_path}")
    print(f"ball={ball_path}")
    print(f"fps={fps:.3f}")
    print(f"toss_gate={toss_gate} ({toss_gate / fps:.3f}s)")
    print(f"hit={hit} ({hit / fps:.3f}s)")
    print(f"bounce={bounce} ({bounce / fps:.3f}s)")
    print(f"saved={output_csv}")


if __name__ == "__main__":
    main()
