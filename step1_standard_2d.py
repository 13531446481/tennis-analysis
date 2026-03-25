import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_ball_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["Frame"]))
            vis = int(float(row["Visibility"]))
            x = float(row["X"])
            y = float(row["Y"])
            rows.append((frame, vis, x, y))

    if not rows:
        raise RuntimeError(f"Empty ball csv: {csv_path}")

    tmax = max(r[0] for r in rows)
    ball_xy = np.full((tmax + 1, 2), np.nan, dtype=np.float32)
    ball_vis = np.zeros((tmax + 1,), dtype=np.uint8)

    for frame, vis, x, y in rows:
        if vis == 1 and x >= 0 and y >= 0:
            ball_xy[frame] = (x, y)
            ball_vis[frame] = 1

    return ball_xy, ball_vis


def homography_from_line20(court20: np.ndarray) -> np.ndarray:
    # image corners -> standard court coordinates (meter)
    # Convention used here:
    # - far baseline has smaller Y
    # - near-camera baseline has larger Y
    # idx2 near-left, idx9 near-right, idx8 far-right, idx6 far-left
    src = np.array([court20[2], court20[9], court20[8], court20[6]], dtype=np.float32)
    dst = np.array([[0.0, 23.77], [8.23, 23.77], [8.23, 0.0], [0.0, 0.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def img_to_world(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    pts = xy.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _meter_to_px(x: float, y: float, scale: int, margin: int):
    u = int(round(margin + x * scale))
    v = int(round(margin + y * scale))
    return u, v


def save_step1_plot(
    out_png: Path,
    hit_world_xy: np.ndarray,
    bounce_world_xy: np.ndarray,
    hit_time_sec: float,
    bounce_time_sec: float,
):
    court_w_m = 8.23
    court_h_m = 23.77
    net_y_m = court_h_m / 2.0
    service_near_y_m = net_y_m - 6.40
    service_far_y_m = net_y_m + 6.40
    center_x_m = court_w_m / 2.0

    scale = 52
    margin = 60
    width = int(round(court_w_m * scale + margin * 2))
    height = int(round(court_h_m * scale + margin * 2))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (34, 94, 42)

    p_tl = _meter_to_px(0.0, 0.0, scale, margin)
    p_tr = _meter_to_px(court_w_m, 0.0, scale, margin)
    p_bl = _meter_to_px(0.0, court_h_m, scale, margin)
    p_br = _meter_to_px(court_w_m, court_h_m, scale, margin)

    cv2.rectangle(canvas, p_tl, p_br, (245, 245, 245), 2)

    net_l = _meter_to_px(0.0, net_y_m, scale, margin)
    net_r = _meter_to_px(court_w_m, net_y_m, scale, margin)
    cv2.line(canvas, net_l, net_r, (180, 180, 180), 2)

    s1_l = _meter_to_px(0.0, service_near_y_m, scale, margin)
    s1_r = _meter_to_px(court_w_m, service_near_y_m, scale, margin)
    s2_l = _meter_to_px(0.0, service_far_y_m, scale, margin)
    s2_r = _meter_to_px(court_w_m, service_far_y_m, scale, margin)
    cv2.line(canvas, s1_l, s1_r, (220, 220, 220), 2)
    cv2.line(canvas, s2_l, s2_r, (220, 220, 220), 2)

    c_top = _meter_to_px(center_x_m, service_near_y_m, scale, margin)
    c_bot = _meter_to_px(center_x_m, service_far_y_m, scale, margin)
    cv2.line(canvas, c_top, c_bot, (220, 220, 220), 2)

    hit_pt = _meter_to_px(float(hit_world_xy[0]), float(hit_world_xy[1]), scale, margin)
    bounce_pt = _meter_to_px(float(bounce_world_xy[0]), float(bounce_world_xy[1]), scale, margin)

    cv2.circle(canvas, hit_pt, 8, (0, 220, 255), -1)
    cv2.circle(canvas, bounce_pt, 8, (255, 180, 0), -1)

    cv2.putText(canvas, f"HIT  t={hit_time_sec:.3f}s", (hit_pt[0] + 10, hit_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"BOUNCE  t={bounce_time_sec:.3f}s", (bounce_pt[0] + 10, bounce_pt[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Standard Singles Court (meter)", (margin, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_png), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write plot image: {out_png}")


def get_ball_xy_at_frame(ball_xy: np.ndarray, ball_vis: np.ndarray, frame_idx: int, search_radius: int = 5):
    if 0 <= frame_idx < len(ball_vis) and ball_vis[frame_idx] == 1:
        return ball_xy[frame_idx].copy(), frame_idx, "exact"

    candidates = []
    for d in range(1, search_radius + 1):
        l = frame_idx - d
        r = frame_idx + d
        if 0 <= l < len(ball_vis) and ball_vis[l] == 1:
            candidates.append((d, l))
        if 0 <= r < len(ball_vis) and ball_vis[r] == 1:
            candidates.append((d, r))

    if candidates:
        _, best = sorted(candidates, key=lambda x: x[0])[0]
        return ball_xy[best].copy(), best, "nearest"

    left = frame_idx - 1
    while left >= 0 and (left >= len(ball_vis) or ball_vis[left] == 0):
        left -= 1
    right = frame_idx + 1
    while right < len(ball_vis) and ball_vis[right] == 0:
        right += 1

    if 0 <= left < len(ball_vis) and 0 <= right < len(ball_vis):
        ratio = (frame_idx - left) / max(right - left, 1)
        xy = ball_xy[left] + ratio * (ball_xy[right] - ball_xy[left])
        return xy.astype(np.float32), frame_idx, "interp"

    raise RuntimeError(f"No valid ball point around frame {frame_idx}")


def get_server_hit_xy_from_feet(players: np.ndarray, frame_idx: int, server_id: int, left_ankle: int = 15, right_ankle: int = 16):
    if frame_idx >= players.shape[0]:
        raise RuntimeError(f"frame {frame_idx} out of range for players with T={players.shape[0]}")
    if server_id < 0 or server_id >= players.shape[1]:
        raise RuntimeError(f"server_id {server_id} out of range")

    p = players[frame_idx, server_id]
    l = p[left_ankle].astype(np.float32)
    r = p[right_ankle].astype(np.float32)

    l_valid = np.all(np.isfinite(l)) and np.linalg.norm(l) > 1e-6
    r_valid = np.all(np.isfinite(r)) and np.linalg.norm(r) > 1e-6

    if l_valid and r_valid:
        return ((l + r) * 0.5).astype(np.float32), frame_idx, "avg_ankles"
    if l_valid:
        return l.copy(), frame_idx, "left_ankle_only"
    if r_valid:
        return r.copy(), frame_idx, "right_ankle_only"

    raise RuntimeError(f"No valid ankle keypoints at frame {frame_idx}, server_id={server_id}")


def resolve_ball_csv(project_root: Path, video_id: str, user_path: str):
    if user_path:
        p = Path(user_path)
        return p if p.is_absolute() else project_root / p

    candidates = [
        project_root / "output" / "ball" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4_pytorch" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4" / f"{video_id}_predict_ball.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Ball csv not found in default locations")


def parse_args():
    parser = argparse.ArgumentParser(description="Step1: map hit/bounce to standard 2D court coordinates")
    parser.add_argument("--video_id", type=str, default="001")
    parser.add_argument("--hit_frame", type=int, required=True)
    parser.add_argument("--bounce_frame", type=int, required=True)
    parser.add_argument(
        "--server_id",
        type=int,
        default=1,
        help="Server player id. Default=1 (near-camera/bottom player). 0=top, 1=bottom",
    )

    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--fps", type=float, default=0.0, help="if 0, try reading from video")

    parser.add_argument("--line_npy", type=str, default="")
    parser.add_argument("--players_npy", type=str, default="output/pose_keypoints/2_keypoints.npy")
    parser.add_argument("--ball_csv", type=str, default="")

    parser.add_argument("--line_ref_frame", type=int, default=-1, help="-1 means use hit_frame")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_plot", type=str, default="", help="Optional output PNG for 2D court visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    line_npy = Path(args.line_npy) if args.line_npy else (project_root / "output" / "line" / f"{args.video_id}.npy")
    if not line_npy.is_absolute():
        line_npy = project_root / line_npy

    players_npy = Path(args.players_npy)
    if not players_npy.is_absolute():
        players_npy = project_root / players_npy

    ball_csv = resolve_ball_csv(project_root, args.video_id, args.ball_csv)

    if not line_npy.exists():
        raise FileNotFoundError(f"line npy not found: {line_npy}")
    if not players_npy.exists():
        raise FileNotFoundError(f"players npy not found: {players_npy}")
    if not ball_csv.exists():
        raise FileNotFoundError(f"ball csv not found: {ball_csv}")

    line = np.load(line_npy)
    if line.ndim != 2 or line.shape[1] != 40:
        raise RuntimeError(f"Unexpected line shape: {line.shape}, expected (T, 40)")

    players = np.load(players_npy)
    if players.ndim != 4 or players.shape[2] < 17 or players.shape[3] != 2:
        raise RuntimeError(f"Unexpected players shape: {players.shape}, expected (T, P, J, 2)")

    ball_xy, ball_vis = load_ball_csv(ball_csv)

    line_ref_frame = args.hit_frame if args.line_ref_frame < 0 else args.line_ref_frame
    line_ref_frame = int(np.clip(line_ref_frame, 0, line.shape[0] - 1))
    court20 = line[line_ref_frame].reshape(20, 2)
    H = homography_from_line20(court20)

    hit_img_xy, hit_used_frame, hit_src = get_server_hit_xy_from_feet(players, args.hit_frame, args.server_id)
    bounce_img_xy, bounce_used_frame, bounce_src = get_ball_xy_at_frame(ball_xy, ball_vis, args.bounce_frame)

    pts_world = img_to_world(H, np.array([hit_img_xy, bounce_img_xy], dtype=np.float32))
    hit_world_xy = pts_world[0]
    bounce_world_xy = pts_world[1]

    fps = float(args.fps)
    if fps <= 0.0 and args.video_path:
        vp = Path(args.video_path)
        if not vp.is_absolute():
            vp = project_root / vp
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    if fps <= 0.0:
        fps = 30.0

    out_dir = project_root / "output" / "step1_2d" / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (out_dir / "step1_2d.json")
    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "step1_2d.csv")
    out_plot = Path(args.out_plot) if args.out_plot else (out_dir / "step1_2d.png")
    if not out_json.is_absolute():
        out_json = project_root / out_json
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv
    if not out_plot.is_absolute():
        out_plot = project_root / out_plot

    result = {
        "video_id": args.video_id,
        "fps": fps,
        "line_ref_frame": int(line_ref_frame),
        "server_id": int(args.server_id),
        "hit": {
            "frame": int(args.hit_frame),
            "used_frame": int(hit_used_frame),
            "source": hit_src,
            "time_sec": float(args.hit_frame / fps),
            "img_xy": [float(hit_img_xy[0]), float(hit_img_xy[1])],
            "world_xy_m": [float(hit_world_xy[0]), float(hit_world_xy[1])],
        },
        "bounce": {
            "frame": int(args.bounce_frame),
            "used_frame": int(bounce_used_frame),
            "source": bounce_src,
            "time_sec": float(args.bounce_frame / fps),
            "img_xy": [float(bounce_img_xy[0]), float(bounce_img_xy[1])],
            "world_xy_m": [float(bounce_world_xy[0]), float(bounce_world_xy[1])],
        },
        "delta_t_sec": float((args.bounce_frame - args.hit_frame) / fps),
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "fps", "line_ref_frame", "server_id",
            "hit_frame", "hit_time_sec", "hit_img_x", "hit_img_y", "hit_world_x_m", "hit_world_y_m", "hit_source",
            "bounce_frame", "bounce_time_sec", "bounce_img_x", "bounce_img_y", "bounce_world_x_m", "bounce_world_y_m", "bounce_source",
            "delta_t_sec",
        ])
        writer.writerow([
            args.video_id,
            f"{fps:.6f}",
            line_ref_frame,
            args.server_id,
            args.hit_frame,
            f"{args.hit_frame / fps:.6f}",
            f"{float(hit_img_xy[0]):.3f}",
            f"{float(hit_img_xy[1]):.3f}",
            f"{float(hit_world_xy[0]):.6f}",
            f"{float(hit_world_xy[1]):.6f}",
            hit_src,
            args.bounce_frame,
            f"{args.bounce_frame / fps:.6f}",
            f"{float(bounce_img_xy[0]):.3f}",
            f"{float(bounce_img_xy[1]):.3f}",
            f"{float(bounce_world_xy[0]):.6f}",
            f"{float(bounce_world_xy[1]):.6f}",
            bounce_src,
            f"{(args.bounce_frame - args.hit_frame) / fps:.6f}",
        ])

    save_step1_plot(
        out_png=out_plot,
        hit_world_xy=hit_world_xy,
        bounce_world_xy=bounce_world_xy,
        hit_time_sec=float(args.hit_frame / fps),
        bounce_time_sec=float(args.bounce_frame / fps),
    )

    print("[STEP1 OK] Standard 2D coordinates generated")
    print(f"  line_npy   : {line_npy}")
    print(f"  players_npy: {players_npy}")
    print(f"  ball_csv   : {ball_csv}")
    print(f"  out_json   : {out_json}")
    print(f"  out_csv    : {out_csv}")
    print(f"  out_plot   : {out_plot}")
    print(f"  hit world  : ({hit_world_xy[0]:.4f}, {hit_world_xy[1]:.4f}) m")
    print(f"  bounce world: ({bounce_world_xy[0]:.4f}, {bounce_world_xy[1]:.4f}) m")
    print(f"  delta t    : {(args.bounce_frame - args.hit_frame) / fps:.4f} s")


if __name__ == "__main__":
    main()
