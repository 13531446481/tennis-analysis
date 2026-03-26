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


def robust_unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.quantile(x, 0.10))
    hi = float(np.quantile(x, 0.90))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


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
    total = int(frames.max() - shift + 1)

    xy = np.full((total, 2), np.nan, dtype=np.float32)
    out_vis = np.zeros((total,), dtype=np.uint8)
    for fr, vi, x, y in zip(frames, vis, xs, ys):
        idx = int(fr - shift)
        if idx < 0 or idx >= total:
            continue
        if vi == 1 and x >= 0 and y >= 0:
            xy[idx] = (x, y)
            out_vis[idx] = 1

    return xy, out_vis


def load_ball_npy(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected npy shape: {arr.shape}")

    xy = arr.astype(np.float32)
    vis = (~((xy[:, 0] == 0) & (xy[:, 1] == 0))).astype(np.uint8)
    xy[vis == 0] = np.nan
    return xy, vis


def interpolate_short_gaps(xy: np.ndarray, vis: np.ndarray, max_gap: int = 2) -> np.ndarray:
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
            ratio = t / float(gap + 1)
            out[a + t] = pa + ratio * (pb - pa)
    return out


def smooth_xy_segmentwise(xy: np.ndarray, k: int = 3) -> np.ndarray:
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
            out[i:j, 0] = moving_average_1d(seg[:, 0], k=k)
            out[i:j, 1] = moving_average_1d(seg[:, 1], k=k)
        i = j
    return out


def detect_turns(y: np.ndarray, eps: float = 0.35, min_gap: int = 3) -> List[Tuple[int, str, float]]:
    vy = np.diff(y, prepend=y[0])
    turns: List[Tuple[int, str, float]] = []
    for t in range(1, len(y) - 1):
        if np.isnan(y[t - 1]) or np.isnan(y[t]) or np.isnan(y[t + 1]):
            continue
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


def compute_ball_over_head(xy: np.ndarray, players: np.ndarray, head_margin: float = 0.0) -> np.ndarray:
    out = np.zeros((len(xy),), dtype=bool)
    upper = min(len(xy), len(players))
    for t in range(upper):
        by = float(xy[t, 1])
        hy = float(players[t, 1, 0, 1])
        if np.isnan(by) or np.isnan(hy):
            continue
        out[t] = (by + head_margin) < hy
    return out


def first_long_visible_start(vis: np.ndarray, min_len: int = 10) -> int:
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


def first_true_cluster(mask: np.ndarray, start: int, max_gap: int = 2, min_len: int = 2) -> Optional[Tuple[int, int]]:
    idx = np.where(mask[start:])[0]
    if len(idx) == 0:
        return None
    seq = idx + start
    cluster = [int(seq[0])]
    for fr in seq[1:]:
        fr = int(fr)
        if fr - cluster[-1] <= max_gap:
            cluster.append(fr)
        else:
            if len(cluster) >= min_len:
                return cluster[0], cluster[-1]
            cluster = [fr]
    if len(cluster) >= min_len:
        return cluster[0], cluster[-1]
    return None


def first_visible_after(vis: np.ndarray, start: int) -> Optional[int]:
    for i in range(max(0, start), len(vis)):
        if int(vis[i]) == 1:
            return i
    return None


def pick_initial_hit(
    turns: List[Tuple[int, str, float]],
    vis: np.ndarray,
    ball_over_head: Optional[np.ndarray],
    fps: float,
    min_turn_score_q: float = 0.35,
    max_follow_sec: float = 0.8,
    over_head_win: int = 2,
) -> Optional[Tuple[int, Optional[int], str]]:
    if not turns:
        return None

    gate = first_long_visible_start(vis, min_len=max(6, int(round(0.20 * fps))))
    cand = [t for t in turns if t[0] >= gate]
    if not cand:
        return None

    search_end = min(len(vis) - 1, gate + int(round(2.2 * fps)))
    reason_suffix = ""
    if ball_over_head is not None and bool(np.any(ball_over_head[gate:])):
        cluster = first_true_cluster(ball_over_head, gate, max_gap=max(2, int(round(0.08 * fps))), min_len=2)
        if cluster is not None:
            oh_start, oh_end = cluster
            local_start = max(gate, oh_start - max(3, int(round(0.20 * fps))))
            local_end = min(len(vis) - 1, oh_end + max(5, int(round(0.45 * fps))))
            cand_local = [t for t in cand if local_start <= t[0] <= local_end]
            if cand_local:
                cand = cand_local
                reason_suffix = "_oh_window"
            else:
                next_vis = first_visible_after(vis, oh_end + 1)
                short_gap_limit = max(3, int(round(0.25 * fps)))
                if next_vis is not None and 1 < (next_vis - oh_end) <= short_gap_limit:
                    return oh_end, None, "over_head_gap_hit"
            search_end = max(search_end, local_end)

    scores = np.array([t[2] for t in cand], dtype=np.float32)
    thr = float(np.quantile(scores, min_turn_score_q)) if len(scores) > 1 else 0.0
    valid = [t for t in cand if t[2] >= thr] or cand

    first = valid[0]
    f0, typ, _ = first
    if typ != "apex":
        return f0, None, f"first_turn_not_apex{reason_suffix}"

    max_follow = max(3, int(round(max_follow_sec * fps)))
    for t in valid:
        if t[1] == "rebound" and t[0] > f0 and (t[0] - f0) <= max_follow:
            return t[0], f0, f"first_turn_is_apex_then_next_rebound{reason_suffix}"

    fallback_rebounds = [t for t in cand if t[1] == "rebound" and f0 < t[0] <= search_end]
    if fallback_rebounds:
        fallback_rebounds.sort(key=lambda item: item[0])
        return fallback_rebounds[0][0], f0, f"fallback_first_rebound_in_window{reason_suffix}"

    early_rebounds = [t for t in turns if t[1] == "rebound" and gate <= t[0] <= search_end]
    if early_rebounds:
        best = max(early_rebounds, key=lambda item: (item[2], -item[0]))
        return best[0], f0, f"fallback_early_rebound{reason_suffix}"

    return None


def pick_next_bounce_after_hit(
    xy: np.ndarray,
    hit_f0: int,
    fps: float,
    min_gap_sec: float = 0.20,
    max_gap_sec: float = 0.65,
    angle_q: float = 0.55,
    min_angle_deg: float = 8.0,
    local_radius: int = 2,
    local_prominence_deg: float = 3.0,
    prefer_y_turn: bool = False,
) -> Optional[int]:
    total = len(xy)
    min_gap = max(2, int(round(min_gap_sec * fps)))
    max_gap = max(min_gap + 1, int(round(max_gap_sec * fps)))
    start = max(1, hit_f0 + min_gap)
    end = min(total - 2, hit_f0 + max_gap)
    if end <= start:
        return None

    cands: List[Tuple[int, float, float, int]] = []
    for t in range(start, end + 1):
        p0 = xy[t - 1]
        p1 = xy[t]
        p2 = xy[t + 1]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue
        v_prev = p1 - p0
        v_next = p2 - p1
        ang = angle_deg(v_prev, v_next)
        y_turn = float((p1[1] - p0[1]) - (p2[1] - p1[1]))
        cands.append((t, ang, y_turn, t - hit_f0))
    if not cands:
        return None

    angles = np.array([item[1] for item in cands], dtype=np.float32)
    pos_y_turn = np.array([max(0.0, item[2]) for item in cands], dtype=np.float32)
    thr_q = float(np.quantile(angles, angle_q)) if len(angles) > 1 else float(angles[0])
    thr = max(float(min_angle_deg), thr_q)

    radius = max(1, int(local_radius))
    prominence_vals: List[float] = []
    expected_gap = 0.38 * fps
    scored: List[Tuple[int, float, float, float, int]] = []
    for i, (t, ang, y_turn, gap) in enumerate(cands):
        left = max(0, i - radius)
        right = min(len(cands), i + radius + 1)
        neigh = [cands[j][1] for j in range(left, right) if j != i]
        base = float(np.median(neigh)) if neigh else 0.0
        prom = float(ang - base)
        prominence_vals.append(prom)
        scored.append((t, ang, y_turn, prom, gap))

    prom_s = robust_unit(np.array(prominence_vals, dtype=np.float32))
    ang_s = robust_unit(angles)
    y_s = robust_unit(pos_y_turn)

    best_frame = None
    best_score = -1e9
    frame_scores: List[Tuple[int, float, float, float, float]] = []
    for i, (t, ang, y_turn, prom, gap) in enumerate(scored):
        if y_turn <= 0:
            continue
        score = (
            0.45 * float(ang_s[i])
            + 0.35 * float(prom_s[i])
            + 0.20 * float(y_s[i])
            - 0.06 * abs(gap - expected_gap)
        )
        if ang >= thr:
            score += 0.20
        if prom >= float(local_prominence_deg):
            score += 0.20
        frame_scores.append((t, score, ang, gap, y_turn))
        if score > best_score:
            best_score = score
            best_frame = t

    ready_gap = max(min_gap, int(round(expected_gap - 0.08 * fps)))
    qualified = [item for item in frame_scores if item[3] >= ready_gap and item[1] >= best_score - 0.25]
    if qualified:
        if prefer_y_turn:
            best_qualified = max(qualified, key=lambda item: (item[1] + 0.015 * item[4], -abs(item[3] - expected_gap), -item[0]))
        else:
            best_qualified = min(qualified, key=lambda item: (abs(item[3] - expected_gap), -item[1], item[0]))
        return int(best_qualified[0])

    if best_frame is not None:
        return int(best_frame)

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
        w = csv.DictWriter(f, fieldnames=["video_id", "hit", "bounce", "hit_sec", "bounce_sec"])
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
    parser.add_argument("--smooth-k", type=int, default=1, help="Smoothing window for hit detection")
    parser.add_argument("--bounce-smooth-k", type=int, default=1, help="Smoothing window for bounce detection")
    parser.add_argument("--interp-max-gap", type=int, default=2, help="Interpolate only short gaps")
    parser.add_argument("--eps", type=float, default=0.35, help="Turning threshold for hit detection")
    parser.add_argument("--turn-q", type=float, default=0.35, help="Quantile threshold for hit turns")
    parser.add_argument("--follow-sec", type=float, default=0.8, help="Follow window after apex for hit")
    parser.add_argument("--bounce-max-sec", type=float, default=0.65, help="Max seconds from hit to bounce")
    parser.add_argument("--bounce-min-frames", type=int, default=6, help="Minimum frames between hit and bounce")
    parser.add_argument("--bounce-angle-q", type=float, default=0.55, help="Quantile threshold for bounce angle")
    parser.add_argument("--bounce-min-angle-deg", type=float, default=8.0, help="Min angle for bounce")
    parser.add_argument("--bounce-local-radius", type=int, default=2, help="Local neighborhood radius")
    parser.add_argument("--bounce-local-prom-deg", type=float, default=3.0, help="Local angle prominence")
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

    xy_i = interpolate_short_gaps(xy, vis, max_gap=max(0, int(args.interp_max_gap)))
    xy_hit = smooth_xy_segmentwise(xy_i, k=max(1, int(args.smooth_k)))
    xy_bounce = smooth_xy_segmentwise(xy_i, k=max(1, int(args.bounce_smooth_k)))
    y_hit = xy_hit[:, 1]
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

    turns = detect_turns(y_hit, eps=float(args.eps), min_gap=max(2, int(round(0.10 * fps))))
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
        xy_bounce,
        hit,
        fps=float(fps),
        min_gap_sec=max(0.12, float(args.bounce_min_frames) / float(fps)),
        max_gap_sec=float(args.bounce_max_sec),
        angle_q=float(args.bounce_angle_q),
        min_angle_deg=float(args.bounce_min_angle_deg),
        local_radius=int(args.bounce_local_radius),
        local_prominence_deg=float(args.bounce_local_prom_deg),
        prefer_y_turn=(reason == "over_head_gap_hit"),
    )
    if bounce is None:
        raise RuntimeError("Failed to detect bounce")

    output_csv = args.out or str(project_root / "output" / "hit_bounce" / f"{video_id}.csv")
    save_result_csv(output_csv, video_id, hit, bounce, fps)

    print(f"video={video_path}")
    print(f"ball={ball_path}")
    print(f"fps={fps:.3f}")
    print(f"turns={len(turns)}")
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
