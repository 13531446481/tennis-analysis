#!/usr/bin/env python3
"""
基于轨迹转折检测击球点：
1) 插值补全不可见点
2) 平滑 y 轨迹
3) 找转折点：
   - apex: 上升->下降（抛球最高点）
   - rebound: 下降->上升（可能是击球或落地反弹）
4) 取“主抛球 apex”之后的第一个强 rebound 作为击球点
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_ball_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data: Dict[int, Tuple[int, float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"Frame", "Visibility", "X", "Y"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"Unexpected csv format: {csv_path}")
        for row in reader:
            fr = int(float(row["Frame"]))
            vis = int(float(row["Visibility"]))
            x = float(row["X"])
            y = float(row["Y"])
            data[fr] = (vis, x, y)

    if not data:
        raise RuntimeError(f"Empty csv: {csv_path}")

    max_frame = max(data.keys())
    xy = np.full((max_frame, 2), np.nan, dtype=np.float32)
    vis = np.zeros((max_frame,), dtype=np.uint8)

    for fr in range(1, max_frame + 1):
        if fr not in data:
            continue
        v, x, y = data[fr]
        idx = fr - 1
        if v == 1 and x >= 0 and y >= 0:
            xy[idx] = [x, y]
            vis[idx] = 1

    return xy, vis


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


def moving_average_1d(x: np.ndarray, k: int = 7) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    ker = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(xp, ker, mode="valid")


def detect_turns(y: np.ndarray, eps: float = 0.35, min_gap: int = 3) -> List[Tuple[int, str, float]]:
    vy = np.diff(y, prepend=y[0])
    turns: List[Tuple[int, str, float]] = []

    for t in range(1, len(y) - 1):
        vp = float(vy[t - 1])
        vc = float(vy[t])
        score = abs(vc - vp)

        # 图像坐标: y小=更高
        # apex: 上升(负) -> 下降(正)
        if vp < -eps and vc > eps:
            turns.append((t, "apex", score))
        # rebound: 下降(正) -> 上升(负)
        elif vp > eps and vc < -eps:
            turns.append((t, "rebound", score))

    # 最小间隔抑制（同类型）
    out: List[Tuple[int, str, float]] = []
    for typ in ("apex", "rebound"):
        seq = [x for x in turns if x[1] == typ]
        seq.sort(key=lambda x: x[0])
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

    out.sort(key=lambda x: x[0])
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
    """ball_over_head[t] = 球是否高于 h1 头部。"""
    T = len(xy)
    out = np.zeros((T,), dtype=bool)
    U = min(T, len(players))
    for t in range(U):
        by = float(xy[t, 1])
        hy = float(players[t, 1, 0, 1])
        if np.isnan(by) or np.isnan(hy):
            continue
        out[t] = (by + head_margin) < hy
    return out


def first_long_visible_start(vis: np.ndarray, min_len: int = 12) -> int:
    best_start = 0
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
    return best_start


def pick_initial_hit(
    turns: List[Tuple[int, str, float]],
    vis: np.ndarray,
    ball_over_head: Optional[np.ndarray],
    fps: float,
    min_turn_score_q: float = 0.45,
    max_follow_sec: float = 1.0,
    over_head_win: int = 2,
) -> Optional[Tuple[int, Optional[int], str]]:
    """
    新逻辑：
    1) 只关心开局阶段的第一个有效转折。
    2) 如果第一个有效转折不是 apex（坠落起点），直接判为 hit。
    3) 如果第一个有效转折是 apex，则取其后最近 rebound 作为 hit。
    """
    if not turns:
        return None

    gate = first_long_visible_start(vis, min_len=max(8, int(round(0.28 * fps))))
    cand = [t for t in turns if t[0] >= gate]
    if not cand:
        cand = turns

    # 击球候选必须在“球高于头”附近帧出现。
    if ball_over_head is not None:
        def above_head_near(frame_idx: int) -> bool:
            l = max(0, frame_idx - over_head_win)
            r = min(len(ball_over_head), frame_idx + over_head_win + 1)
            return bool(np.any(ball_over_head[l:r]))

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
    """
    网球规则约束：击球后下一次明显方向转折应为落地（bounce）。
    采用局部角度显著性：
    - 在时间窗内计算每个中点的三点夹角
    - 若该角度显著高于附近点（邻域中位数 + prominence），则判为转折
    - 取第一个显著转折点
    """
    T = len(xy_s)
    min_gap = max(1, int(round(min_gap_sec * fps)))
    max_gap = max(min_gap + 1, int(round(max_gap_sec * fps)))

    st = max(1, hit_f0 + min_gap)
    ed = min(T - 2, hit_f0 + max_gap)
    if ed <= st:
        return None

    # 候选项: (frame_index, angle_deg)
    cands: List[Tuple[int, float]] = []
    for t in range(st, ed + 1):
        p0 = xy_s[t - 1]
        p1 = xy_s[t]
        p2 = xy_s[t + 1]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue

        v_prev = p1 - p0
        v_next = p2 - p1
        ang = angle_deg(v_prev, v_next)
        cands.append((t, ang))

    if not cands:
        return None

    angles = np.array([a for _, a in cands], dtype=np.float32)
    thr_q = float(np.quantile(angles, angle_q)) if len(angles) > 1 else float(angles[0])
    thr = max(float(min_angle_deg), thr_q)

    # 局部显著性：角度比邻域明显更大（不要求局部峰值）
    sig: List[Tuple[int, float, float]] = []  # (frame, angle, prominence)
    r = max(1, int(local_radius))
    for i, (t, a) in enumerate(cands):
        l = max(0, i - r)
        rr = min(len(cands), i + r + 1)
        neigh = [cands[j][1] for j in range(l, rr) if j != i]
        local_base = float(np.median(neigh)) if neigh else 0.0
        prom = float(a - local_base)
        if a >= thr and prom >= float(local_prominence_deg):
            sig.append((t, a, prom))

    if sig:
        # 取第一个显著转折点（强调“第一个转折”）
        sig.sort(key=lambda x: x[0])
        return int(sig[0][0])

    # 回退：没有明显显著峰时，取全窗口角度最大者
    best = max(cands, key=lambda x: x[1])
    return int(best[0])


def resolve_ball_csv(project_root: Path, video_id: str, ball_csv_arg: str) -> Path:
    if ball_csv_arg:
        p = Path(ball_csv_arg)
        return p if p.is_absolute() else project_root / p

    candidates = [
        project_root / "output" / "ball" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4_pytorch" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4" / f"{video_id}_predict_ball.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No ball csv found for {video_id}")


def save_result(
    path: Path,
    video_id: str,
    toss_f0: int,
    hit_f0: int,
    fps: float,
    bounce_f0: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "toss_apex",
                "hit",
                "bounce",
                "toss_apex_sec",
                "hit_sec",
                "bounce_sec",
                "gap_frames",
                "gap_sec",
                "hit_to_bounce_frames",
                "hit_to_bounce_sec",
            ],
        )
        w.writeheader()
        hit_to_bounce_frames = (int(bounce_f0 - hit_f0) if bounce_f0 is not None else "")
        hit_to_bounce_sec = (round(float((bounce_f0 - hit_f0) / fps), 3) if bounce_f0 is not None else "")
        w.writerow(
            {
                "video_id": video_id,
                "toss_apex": toss_f0 + 1,
                "hit": hit_f0 + 1,
                "bounce": (bounce_f0 + 1) if bounce_f0 is not None else "",
                "toss_apex_sec": round((toss_f0 + 1) / fps, 3),
                "hit_sec": round((hit_f0 + 1) / fps, 3),
                "bounce_sec": (round((bounce_f0 + 1) / fps, 3) if bounce_f0 is not None else ""),
                "gap_frames": int(hit_f0 - toss_f0),
                "gap_sec": round((hit_f0 - toss_f0) / fps, 3),
                "hit_to_bounce_frames": hit_to_bounce_frames,
                "hit_to_bounce_sec": hit_to_bounce_sec,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect hit point from toss-then-hit turning logic")
    parser.add_argument("--video-id", default="001", help="Video ID")
    parser.add_argument("--ball-csv", default="", help="Optional ball csv path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS")
    parser.add_argument("--smooth-k", type=int, default=7, help="Smoothing window")
    parser.add_argument("--eps", type=float, default=0.35, help="Turning threshold")
    parser.add_argument("--turn-q", type=float, default=0.45, help="Quantile threshold for valid turns")
    parser.add_argument("--follow-sec", type=float, default=1.0, help="Max seconds from apex to rebound")
    parser.add_argument("--bounce-max-sec", type=float, default=0.63, help="Max seconds from hit to bounce candidate")
    parser.add_argument("--bounce-min-frames", type=int, default=6, help="Minimum frames between hit and bounce")
    parser.add_argument("--bounce-angle-q", type=float, default=0.60, help="Quantile threshold for large direction turn")
    parser.add_argument("--bounce-min-angle-deg", type=float, default=10.0, help="Minimum turning angle for bounce")
    parser.add_argument("--bounce-local-radius", type=int, default=3, help="Neighbor radius for local angle prominence")
    parser.add_argument("--bounce-local-prom-deg", type=float, default=6.0, help="Required local angle prominence (deg)")
    parser.add_argument("--players", default="", help="Optional players path, default output/pose_keypoints/2_keypoints.npy")
    parser.add_argument("--head-margin", type=float, default=0.0, help="Ball-over-head margin in pixels")
    parser.add_argument("--over-head-win", type=int, default=2, help="Neighbor frames window for over-head check")
    parser.add_argument("--out", default="", help="Output csv path")
    args = parser.parse_args()

    if args.smooth_k < 1:
        raise ValueError("--smooth-k must be >= 1")
    if args.smooth_k % 2 == 0:
        args.smooth_k += 1

    project_root = Path(__file__).resolve().parents[1]
    ball_csv = resolve_ball_csv(project_root, args.video_id, args.ball_csv)

    xy, vis = load_ball_csv(ball_csv)
    xy_i = interpolate_all_gaps(xy, vis)
    x_s = moving_average_1d(xy_i[:, 0], k=int(args.smooth_k))
    y_s = moving_average_1d(xy_i[:, 1], k=int(args.smooth_k))
    xy_s = np.stack([x_s, y_s], axis=1)

    # 计算球是否高于头部（h1）
    ball_over_head: Optional[np.ndarray] = None
    players_path = Path(args.players) if args.players else (project_root / "output" / "pose_keypoints" / "2_keypoints.npy")
    if not players_path.is_absolute():
        players_path = project_root / players_path
    if players_path.is_file():
        try:
            players = load_players(players_path)
            ball_over_head = compute_ball_over_head(xy_i, players, head_margin=float(args.head_margin))
        except Exception as e:
            print(f"[WARN] over-head constraint disabled: {e}")
            ball_over_head = None
    else:
        print("[WARN] players file not found, over-head constraint disabled")

    turns = detect_turns(y_s, eps=float(args.eps), min_gap=max(2, int(round(0.12 * args.fps))))
    picked = pick_initial_hit(
        turns,
        vis,
        ball_over_head,
        fps=float(args.fps),
        min_turn_score_q=float(args.turn_q),
        max_follow_sec=float(args.follow_sec),
        over_head_win=max(0, int(args.over_head_win)),
    )
    if picked is None:
        raise RuntimeError("Failed to detect hit under initial-turn + ball-over-head constraints")

    hit_f0, toss_f0, reason = picked
    bounce_f0 = pick_next_bounce_after_hit(
        xy_s,
        hit_f0,
        fps=float(args.fps),
        min_gap_sec=max(0.08, float(args.bounce_min_frames) / float(args.fps)),
        max_gap_sec=float(args.bounce_max_sec),
        angle_q=float(args.bounce_angle_q),
        min_angle_deg=float(args.bounce_min_angle_deg),
        local_radius=int(args.bounce_local_radius),
        local_prominence_deg=float(args.bounce_local_prom_deg),
    )

    out_path = Path(args.out) if args.out else project_root / "output" / "ball" / f"hit_from_turns_{args.video_id}.csv"
    if not out_path.is_absolute():
        out_path = project_root / out_path

    save_result(
        out_path,
        args.video_id,
        toss_f0 if toss_f0 is not None else hit_f0,
        hit_f0,
        float(args.fps),
        bounce_f0=bounce_f0,
    )

    print(f"ball_csv={ball_csv}")
    print(f"frames={len(xy)} visible={int(vis.sum())}")
    print(f"turns={len(turns)}")
    print(
        f"bounce_max_sec={float(args.bounce_max_sec):.3f} "
        f"bounce_min_frames={int(args.bounce_min_frames)} "
        f"bounce_angle_q={float(args.bounce_angle_q):.3f} "
        f"bounce_min_angle_deg={float(args.bounce_min_angle_deg):.1f} "
        f"bounce_local_radius={int(args.bounce_local_radius)} "
        f"bounce_local_prom_deg={float(args.bounce_local_prom_deg):.1f}"
    )
    if ball_over_head is not None:
        print(f"over_head_frames={int(np.sum(ball_over_head))}")
    if toss_f0 is not None:
        print(f"toss_apex={toss_f0 + 1} ({(toss_f0 + 1) / args.fps:.3f}s)")
    else:
        print("toss_apex=None (first turn treated as hit)")
    print(f"hit={hit_f0 + 1} ({(hit_f0 + 1) / args.fps:.3f}s)")
    if bounce_f0 is not None:
        print(f"bounce={bounce_f0 + 1} ({(bounce_f0 + 1) / args.fps:.3f}s)")
        print(f"hit_to_bounce={bounce_f0 - hit_f0} frames ({(bounce_f0 - hit_f0) / args.fps:.3f}s)")
    else:
        print("bounce=None")
    if toss_f0 is not None:
        print(f"gap={hit_f0 - toss_f0} frames ({(hit_f0 - toss_f0) / args.fps:.3f}s)")
    print(f"reason={reason}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
