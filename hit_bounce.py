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


def interpolate_all_gaps(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    out = xy.copy().astype(np.float32)
    idx = np.arange(len(out), dtype=np.float32)
    valid_idx = idx[vis.astype(bool)]
    if len(valid_idx) < 2:
        raise RuntimeError("Not enough visible points to interpolate")

    for dim in range(2):
        vals = out[vis.astype(bool), dim]
        out[:, dim] = np.interp(idx, valid_idx, vals)
    return out


def smooth_xy(xy: np.ndarray, k: int = 5) -> np.ndarray:
    x = moving_average_1d(xy[:, 0], k)
    y = moving_average_1d(xy[:, 1], k)
    return np.stack([x, y], axis=1)


def smooth_xy_segmentwise(xy: np.ndarray, k: int = 3) -> np.ndarray:
    """Apply smoothing only within contiguous valid segments.

    This avoids blending across long missing intervals, which can bend
    trajectory turns and hurt hit/bounce timing.
    """
    if k <= 1:
        return xy.astype(np.float32)

    out = xy.astype(np.float32).copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isnan(out[i]).any():
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]).any():
            j += 1

        seg = out[i:j]
        if len(seg) >= k:
            xs = moving_average_1d(seg[:, 0], k=k)
            ys = moving_average_1d(seg[:, 1], k=k)
            out[i:j, 0] = xs
            out[i:j, 1] = ys

        i = j

    return out


def detect_turns(y: np.ndarray, eps: float = 0.35, min_gap: int = 3) -> List[Tuple[int, str, float]]:
    vy = np.diff(y, prepend=y[0])
    turns: List[Tuple[int, str, float]] = []

    for t in range(1, len(y) - 1):
        vp = float(vy[t - 1])
        vc = float(vy[t])
        score = abs(vc - vp)

        if vp < -eps and vc > eps:
            turns.append((t, "apex", score))
        elif vp > eps and vc < -eps:
            turns.append((t, "rebound", score))

    out: List[Tuple[int, str, float]] = []
    for typ in ("apex", "rebound"):
        seq = [item for item in turns if item[1] == typ]
        seq.sort(key=lambda item: item[0])
        keep: List[Tuple[int, str, float]] = []
        for item in seq:
            if not keep:
                keep.append(item)
                continue
            if item[0] - keep[-1][0] < min_gap:
                if item[2] > keep[-1][2]:
                    keep[-1] = item
            else:
                keep.append(item)
        out.extend(keep)

    out.sort(key=lambda item: item[0])
    return out


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def load_players(players_path: Path) -> np.ndarray:
    arr = np.load(players_path)
    if arr.ndim != 4 or arr.shape[1] < 2 or arr.shape[2] < 1 or arr.shape[3] != 2:
        raise RuntimeError(f"Unexpected players shape: {arr.shape}")
    return arr.astype(np.float32)


def compute_ball_over_head(
    xy: np.ndarray,
    players: np.ndarray,
    head_margin: float = 0.0,
) -> np.ndarray:
    out = np.zeros((len(xy),), dtype=bool)
    upper = min(len(xy), len(players))
    for t in range(upper):
        by = float(xy[t, 1])
        hy = float(players[t, 1, 0, 1])
        if np.isnan(by) or np.isnan(hy):
            continue
        out[t] = (by + head_margin) < hy
    return out


def first_long_visible_start(vis: np.ndarray, min_len: int = 12) -> int:
    run = 0
    run_start = 0
    for i, v in enumerate(vis.tolist()):
        if int(v) == 1:
            if run == 0:
                run_start = i
            run += 1
            if run >= min_len:
                return run_start
        else:
            run = 0
    return 0


def pick_initial_hit(
    turns: List[Tuple[int, str, float]],
    vis: np.ndarray,
    ball_over_head: Optional[np.ndarray],
    fps: float,
    min_turn_score_q: float = 0.45,
    max_follow_sec: float = 1.0,
    over_head_win: int = 2,
) -> Optional[Tuple[int, Optional[int], str]]:
    if not turns:
        return None

    gate = first_long_visible_start(vis, min_len=max(8, int(round(0.28 * fps))))
    cand = [t for t in turns if t[0] >= gate]
    if not cand:
        cand = turns

    if ball_over_head is not None:
        def above_head_near(frame_idx: int) -> bool:
            left = max(0, frame_idx - over_head_win)
            right = min(len(ball_over_head), frame_idx + over_head_win + 1)
            return bool(np.any(ball_over_head[left:right]))

        cand_oh = [t for t in cand if above_head_near(t[0])]
        if cand_oh:
            cand = cand_oh
        else:
            return None

    scores = np.array([t[2] for t in cand], dtype=np.float32)
    thr = float(np.quantile(scores, min_turn_score_q)) if len(scores) > 0 else 0.0
    valid = [t for t in cand if t[2] >= thr] or cand

    first = valid[0]
    f0, typ, _ = first
    if typ != "apex":
        return f0, None, "first_turn_not_apex"

    max_follow = max(3, int(round(max_follow_sec * fps)))
    for t in valid:
        if t[1] == "rebound" and t[0] > f0 and (t[0] - f0) <= max_follow:
            return t[0], f0, "first_turn_is_apex_then_next_rebound"

    for t in cand:
        if t[1] == "rebound" and t[0] > f0:
            return t[0], f0, "fallback_first_rebound_after_apex"

    return None


def pick_next_bounce_after_hit(
    xy_s: np.ndarray,
    hit_f0: int,
    fps: float,
    min_gap_sec: float = 0.08,
    max_gap_sec: float = 0.63,
    angle_q: float = 0.60,
    min_angle_deg: float = 10.0,
    local_radius: int = 3,
    local_prominence_deg: float = 6.0,
) -> Optional[int]:
    total = len(xy_s)
    min_gap = max(1, int(round(min_gap_sec * fps)))
    max_gap = max(min_gap + 1, int(round(max_gap_sec * fps)))

    start = max(1, hit_f0 + min_gap)
    end = min(total - 2, hit_f0 + max_gap)
    if end <= start:
        return None

    cands: List[Tuple[int, float]] = []
    for t in range(start, end + 1):
        p0 = xy_s[t - 1]
        p1 = xy_s[t]
        p2 = xy_s[t + 1]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue

        v_prev = p1 - p0
        v_next = p2 - p1
        cands.append((t, angle_deg(v_prev, v_next)))

    if not cands:
        return None

    angles = np.array([angle for _, angle in cands], dtype=np.float32)
    thr_q = float(np.quantile(angles, angle_q)) if len(angles) > 1 else float(angles[0])
    thr = max(float(min_angle_deg), thr_q)

    sig: List[Tuple[int, float, float]] = []
    radius = max(1, int(local_radius))
    for i, (t, angle) in enumerate(cands):
        left = max(0, i - radius)
        right = min(len(cands), i + radius + 1)
        neigh = [cands[j][1] for j in range(left, right) if j != i]
        local_base = float(np.median(neigh)) if neigh else 0.0
        prominence = float(angle - local_base)
        if angle >= thr and prominence >= float(local_prominence_deg):
            sig.append((t, angle, prominence))

    if sig:
        sig.sort(key=lambda item: item[0])
        return int(sig[0][0])

    return int(max(cands, key=lambda item: item[1])[0])


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
        project_root / "output" / "ball" / f"{video_id}_predict_ball.csv",
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
    parser.add_argument("--smooth-k", type=int, default=3, help="Smoothing window for trajectory")
    parser.add_argument("--interp-max-gap", type=int, default=2, help="Deprecated: kept for compatibility")
    parser.add_argument("--eps", type=float, default=0.35, help="Turning threshold for hit detection")
    parser.add_argument("--turn-q", type=float, default=0.45, help="Quantile threshold for hit turns")
    parser.add_argument("--follow-sec", type=float, default=1.0, help="Follow window after apex for hit")
    parser.add_argument("--bounce-max-sec", type=float, default=0.63, help="Max seconds from hit to bounce")
    parser.add_argument("--bounce-min-frames", type=int, default=6, help="Minimum frames between hit and bounce")
    parser.add_argument("--bounce-angle-q", type=float, default=0.60, help="Quantile threshold for bounce angle")
    parser.add_argument("--bounce-min-angle-deg", type=float, default=10.0, help="Min angle for bounce")
    parser.add_argument("--bounce-local-radius", type=int, default=3, help="Local neighborhood radius")
    parser.add_argument("--bounce-local-prom-deg", type=float, default=6.0, help="Local angle prominence")
    parser.add_argument("--head-margin", type=float, default=0.0, help="Ball-over-head margin")
    parser.add_argument("--over-head-win", type=int, default=2, help="Neighbor frames for over-head check")
    parser.add_argument("--out", default="", help="Optional output csv path")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    script_dir = Path(__file__).resolve().parent
    if (script_dir / "output").exists() and (script_dir / "videos").exists():
        project_root = script_dir
    elif (script_dir.parent / "output").exists() and (script_dir.parent / "videos").exists():
        project_root = script_dir.parent
    else:
        project_root = script_dir
    video_id = video_path.stem
    ball_path = resolve_ball_path(project_root, video_id, args.ball)

    if ball_path.endswith(".csv"):
        xy, vis = load_ball_csv(str(Path(ball_path)))
    elif ball_path.endswith(".npy"):
        xy, vis = load_ball_npy(ball_path)
    else:
        raise RuntimeError(f"Unsupported ball file: {ball_path}")

    xy_i = interpolate_all_gaps(xy, vis)
    xy_s = smooth_xy(xy_i, k=max(1, int(args.smooth_k)))
    y_s = xy_s[:, 1]
    fps = get_fps(str(video_path))

    players_path = Path(args.players) if args.players else (project_root / "output" / "pose_keypoints" / video_id / "2_keypoints.npy")
    if not players_path.is_absolute():
        players_path = project_root / players_path

    ball_over_head: Optional[np.ndarray] = None
    if players_path.is_file():
        try:
            players = load_players(players_path)
            ball_over_head = compute_ball_over_head(xy_i, players, head_margin=float(args.head_margin))
        except Exception as e:
            print(f"[WARN] over-head constraint disabled due to players issue: {e}")
    else:
        print("[WARN] players file not found, over-head constraint disabled")

    turns = detect_turns(y_s, eps=float(args.eps), min_gap=max(2, int(round(0.12 * fps))))
    picked = pick_initial_hit(
        turns,
        vis,
        ball_over_head,
        fps=float(fps),
        min_turn_score_q=float(args.turn_q),
        max_follow_sec=float(args.follow_sec),
        over_head_win=max(0, int(args.over_head_win)),
    )
    if picked is None:
        raise RuntimeError("Failed to detect hit")

    hit, toss_f0, reason = picked
    bounce = pick_next_bounce_after_hit(
        xy_s,
        hit,
        fps=float(fps),
        min_gap_sec=max(0.08, float(args.bounce_min_frames) / float(fps)),
        max_gap_sec=float(args.bounce_max_sec),
        angle_q=float(args.bounce_angle_q),
        min_angle_deg=float(args.bounce_min_angle_deg),
        local_radius=int(args.bounce_local_radius),
        local_prominence_deg=float(args.bounce_local_prom_deg),
    )
    if bounce is None:
        raise RuntimeError("Failed to detect bounce")

    output_csv = args.out
    if not output_csv:
        output_csv = str(project_root / "output" / "hit_bounce" / f"{video_id}.csv")

    save_result_csv(output_csv, video_id, hit, bounce, fps)

    print(f"video={video_path}")
    print(f"ball={ball_path}")
    print(f"fps={fps:.3f}")
    if toss_f0 is not None:
        print(f"toss_apex={toss_f0} ({toss_f0 / fps:.3f}s)")
    else:
        print("toss_apex=None")
    print(f"hit={hit} ({hit / fps:.3f}s)")
    print(f"bounce={bounce} ({bounce / fps:.3f}s)")
    print(f"reason={reason}")
    print(f"saved={output_csv}")


if __name__ == "__main__":
    main()
