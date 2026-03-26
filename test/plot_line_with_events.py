#!/usr/bin/env python3
"""在视频首帧上绘制球轨迹连线，并高亮事件帧。"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def load_players(players_path: Path) -> np.ndarray:
    arr = np.load(players_path)
    if arr.ndim != 4 or arr.shape[1] < 2 or arr.shape[2] < 1 or arr.shape[3] != 2:
        raise RuntimeError(f"Unexpected players shape: {arr.shape}")
    return arr.astype(np.float32)


def compute_ball_over_head(pts, players: np.ndarray, head_margin: float = 0.0):
    out = {}
    if players is None:
        return out
    tmax = players.shape[0]
    for fr, x, y in pts:
        t = fr
        if t < 0 or t >= tmax:
            continue
        hy = float(players[t, 1, 0, 1])
        if np.isnan(hy):
            continue
        out[fr] = bool(float(y) + float(head_margin) < hy)
    return out


def load_ball(csv_path: Path) -> Dict[int, Tuple[int, float, float]]:
    out: Dict[int, Tuple[int, float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"Frame", "Visibility", "X", "Y"}
        if rd.fieldnames is None or not need.issubset(set(rd.fieldnames)):
            raise RuntimeError(f"Unexpected csv format: {csv_path}")
        for row in rd:
            fr = int(float(row["Frame"]))
            vis = int(float(row["Visibility"]))
            x = float(row["X"])
            y = float(row["Y"])
            out[fr] = (vis, x, y)
    return out


def draw_label(frame, text: str, x: int, y: int, color):
    tx, ty = x + 8, y - 8
    if ty < 20:
        ty = y + 20
    cv2.rectangle(frame, (tx - 2, ty - 15), (tx + 210, ty + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def interp_event_xy(pts, target_fr: int):
    """若目标帧不可见，使用相邻可见点做线性插值估计位置。"""
    if not pts:
        return None
    frames = [p[0] for p in pts]
    if target_fr < frames[0] or target_fr > frames[-1]:
        return None

    prev_p = None
    next_p = None
    for p in pts:
        if p[0] <= target_fr:
            prev_p = p
        if p[0] >= target_fr and next_p is None:
            next_p = p

    if prev_p is None or next_p is None:
        return None
    if prev_p[0] == next_p[0]:
        return prev_p[1], prev_p[2]

    f0, x0, y0 = prev_p
    f1, x1, y1 = next_p
    r = (target_fr - f0) / float(f1 - f0)
    xi = int(round(x0 + r * (x1 - x0)))
    yi = int(round(y0 + r * (y1 - y0)))
    return xi, yi


def split_visible_segments(points, max_inner_gap: int = 2):
    if not points:
        return []
    segs = []
    cur = [points[0]]
    for i in range(1, len(points)):
        pf = points[i - 1][0]
        cf = points[i][0]
        if cf - pf <= max_inner_gap:
            cur.append(points[i])
        else:
            segs.append(cur)
            cur = [points[i]]
    segs.append(cur)
    return segs


def select_segment(points, mode: str, min_len: int, max_inner_gap: int):
    if mode == "off":
        return points, [points]
    segs = split_visible_segments(points, max_inner_gap=max_inner_gap)
    if not segs:
        return [], []
    if mode == "all-long":
        out = []
        keep = [s for s in segs if len(s) >= min_len]
        for s in keep:
            out.extend(s)
        return out, segs
    best = max(segs, key=len)
    return best, segs


def denoise_flyaway_points(points, max_gap_frames: int = 6, z_thresh: float = 4.0, min_err_px: float = 18.0):
    n = len(points)
    if n < 3:
        return points[:], []

    errs = []
    idx_map = []
    for i in range(1, n - 1):
        f0, x0, y0 = points[i - 1]
        f1, x1, y1 = points[i]
        f2, x2, y2 = points[i + 1]
        if not (f0 < f1 < f2):
            continue
        if (f1 - f0) > max_gap_frames or (f2 - f1) > max_gap_frames:
            continue
        r = (f1 - f0) / float(f2 - f0)
        px = x0 + r * (x2 - x0)
        py = y0 + r * (y2 - y0)
        err = float(np.hypot(x1 - px, y1 - py))
        errs.append(err)
        idx_map.append(i)

    if not errs:
        return points[:], []

    e = np.array(errs, dtype=np.float32)
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)))
    sigma = 1.4826 * mad
    thr = max(min_err_px, med + z_thresh * sigma)

    remove_idx = set()
    for j, i in enumerate(idx_map):
        if errs[j] > thr:
            remove_idx.add(i)

    keep = []
    removed = []
    for i, p in enumerate(points):
        if i in remove_idx:
            removed.append(p)
        else:
            keep.append(p)
    return keep, removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="001")
    ap.add_argument("--hit", type=int, default=197)
    ap.add_argument("--bounce", type=int, default=255)
    ap.add_argument("--segment-mode", choices=["longest", "all-long", "off"], default="longest")
    ap.add_argument("--segment-gap", type=int, default=2)
    ap.add_argument("--segment-min-len", type=int, default=20)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--z-thresh", type=float, default=4.0)
    ap.add_argument("--min-err", type=float, default=18.0)
    ap.add_argument("--players", default="", help="Optional players npy for over-head check")
    ap.add_argument("--head-margin", type=float, default=0.0, help="Ball-over-head margin in pixels")
    ap.add_argument("--raw-only", action="store_true", help="Draw raw visible points only (no segment/denoise)")
    ap.add_argument("--show-frame-labels", action="store_true", help="Draw frame index label near each point")
    ap.add_argument("--label-step", type=int, default=1, help="Frame label step (1 means label every point)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame (inclusive) for plotting")
    ap.add_argument("--end-frame", type=int, default=-1, help="End frame (inclusive), -1 means no upper limit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    video = root / "videos" / f"{args.video_id}.mp4"
    ball_csv = root / "output" / "ball" / f"{args.video_id}_predict_ball.csv"
    if not video.is_file():
        raise FileNotFoundError(video)
    if not ball_csv.is_file():
        raise FileNotFoundError(ball_csv)

    ball = load_ball(ball_csv)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    start_fr = max(0, int(args.start_frame))
    end_fr = int(args.end_frame)

    pts_raw = []
    for fr in sorted(ball.keys()):
        if fr < start_fr:
            continue
        if end_fr >= 0 and fr > end_fr:
            continue
        vis, x, y = ball[fr]
        if vis == 1 and x >= 0 and y >= 0:
            pts_raw.append((fr, int(round(x)), int(round(y))))

    if args.raw_only:
        pts_seg = pts_raw
        segs = [pts_raw] if pts_raw else []
        pts = pts_raw
        removed = []
    else:
        pts_seg, segs = select_segment(
            pts_raw,
            mode=str(args.segment_mode),
            min_len=int(args.segment_min_len),
            max_inner_gap=int(args.segment_gap),
        )
        if not pts_seg:
            raise RuntimeError("No points selected after segment filtering")

        if args.no_denoise:
            pts = pts_seg
            removed = []
        else:
            pts, removed = denoise_flyaway_points(
                pts_seg,
                max_gap_frames=int(args.max_gap),
                z_thresh=float(args.z_thresh),
                min_err_px=float(args.min_err),
            )
        if not pts:
            raise RuntimeError("No points left after denoise")

    for i in range(1, len(pts)):
        _, x0, y0 = pts[i - 1]
        _, x1, y1 = pts[i]
        cv2.line(frame, (x0, y0), (x1, y1), (0, 200, 0), 2, cv2.LINE_AA)

    for fr, x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    if args.show_frame_labels:
        step = max(1, int(args.label_step))
        for i, (fr, x, y) in enumerate(pts):
            if i % step != 0:
                continue
            cv2.putText(
                frame,
                f"{fr}",
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    pts_map = {fr: (x, y) for fr, x, y in pts}

    players_path = Path(args.players) if args.players else (root / "output" / "pose_keypoints" / "2_keypoints.npy")
    if not players_path.is_absolute():
        players_path = root / players_path
    players = None
    if players_path.is_file():
        try:
            players = load_players(players_path)
        except Exception:
            players = None

    over_head = compute_ball_over_head(pts, players, head_margin=float(args.head_margin))

    # 在预处理描点上额外标注“球高于头(h1)”状态
    for fr, x, y in pts:
        if over_head.get(fr, False):
            cv2.circle(frame, (x, y), 6, (255, 0, 255), 1, cv2.LINE_AA)

    event_colors = {
        args.hit: (0, 255, 255),
        args.bounce: (255, 255, 0),
    }
    for efr, c in event_colors.items():
        if efr in pts_map:
            xi, yi = pts_map[efr]
            cv2.circle(frame, (xi, yi), 8, c, 2)
            tag = "HIT" if efr == args.hit else "BOUNCE"
            oh = over_head.get(efr, None)
            oh_text = "over_head=NA" if oh is None else f"over_head={int(bool(oh))}"
            draw_label(frame, f"{tag} f{efr} ({xi},{yi}) {oh_text}", xi, yi, c)
        else:
            # 不可见帧：尝试用相邻可见点插值估计位置并高亮
            est = interp_event_xy(pts, efr)
            if est is not None:
                xi, yi = est
                cv2.circle(frame, (xi, yi), 8, c, 2)
                cv2.line(frame, (xi - 8, yi), (xi + 8, yi), c, 2, cv2.LINE_AA)
                cv2.line(frame, (xi, yi - 8), (xi, yi + 8), c, 2, cv2.LINE_AA)
                tag = "HIT" if efr == args.hit else "BOUNCE"
                oh = over_head.get(efr, None)
                oh_text = "over_head=NA" if oh is None else f"over_head={int(bool(oh))}"
                draw_label(frame, f"{tag} f{efr} interp({xi},{yi}) {oh_text}", xi, yi, c)

    oh_count = int(sum(1 for _, v in over_head.items() if v))

    cv2.rectangle(frame, (10, 10), (1040, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"video={args.video_id} red=ball green=line yellow=hit({args.hit}) cyan=bounce({args.bounce}) magenta_ring=over_head", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    mode_text = "raw-only" if args.raw_only else "segment+denoise"
    cv2.putText(frame, f"preprocess[{mode_text}]: raw={len(pts_raw)} seg={len(pts_seg)} keep={len(pts)} rm={len(removed)} segs={len(segs)}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"over_head_true={oh_count}/{len(pts)}  players={'yes' if players is not None else 'no'}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 1, cv2.LINE_AA)

    out = Path(args.out) if args.out else root / "output" / "test" / f"{args.video_id}_line_hit_bounce_preprocessed.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"Failed to save {out}")

    print(f"raw={len(pts_raw)} seg={len(pts_seg)} keep={len(pts)} removed={len(removed)} segs={len(segs)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
