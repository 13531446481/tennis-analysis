import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def solve_initial_velocity(
    hit_xyz: np.ndarray,
    bounce_xyz: np.ndarray,
    delta_t: float,
    g: float,
) -> np.ndarray:
    if delta_t <= 0:
        raise ValueError(f"delta_t must be > 0, got {delta_t}")

    vx0 = (bounce_xyz[0] - hit_xyz[0]) / delta_t
    vy0 = (bounce_xyz[1] - hit_xyz[1]) / delta_t
    vz0 = (bounce_xyz[2] - hit_xyz[2] + 0.5 * g * delta_t * delta_t) / delta_t
    return np.array([vx0, vy0, vz0], dtype=np.float64)


def sample_trajectory(
    hit_xyz: np.ndarray,
    v0_xyz: np.ndarray,
    g: float,
    duration: float,
    n_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, duration, n_points, dtype=np.float64)
    xyz = np.zeros((n_points, 3), dtype=np.float64)

    xyz[:, 0] = hit_xyz[0] + v0_xyz[0] * t
    xyz[:, 1] = hit_xyz[1] + v0_xyz[1] * t
    xyz[:, 2] = hit_xyz[2] + v0_xyz[2] * t - 0.5 * g * t * t
    return t, xyz


def save_xy_plot_png(
    out_png: Path,
    hit_xy: np.ndarray,
    bounce_xy: np.ndarray,
    traj_xy: np.ndarray,
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

    def m2p(x: float, y: float) -> tuple[int, int]:
        u = int(round(margin + x * scale))
        v = int(round(margin + y * scale))
        return u, v

    p_tl = m2p(0.0, 0.0)
    p_br = m2p(court_w_m, court_h_m)
    cv2.rectangle(canvas, p_tl, p_br, (245, 245, 245), 2)

    cv2.line(canvas, m2p(0.0, net_y_m), m2p(court_w_m, net_y_m), (180, 180, 180), 2)
    cv2.line(canvas, m2p(0.0, service_near_y_m), m2p(court_w_m, service_near_y_m), (220, 220, 220), 2)
    cv2.line(canvas, m2p(0.0, service_far_y_m), m2p(court_w_m, service_far_y_m), (220, 220, 220), 2)
    cv2.line(canvas, m2p(center_x_m, service_near_y_m), m2p(center_x_m, service_far_y_m), (220, 220, 220), 2)

    pts = [m2p(float(x), float(y)) for x, y in traj_xy]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (80, 180, 255), 2, cv2.LINE_AA)

    hit_pt = m2p(float(hit_xy[0]), float(hit_xy[1]))
    bounce_pt = m2p(float(bounce_xy[0]), float(bounce_xy[1]))
    cv2.circle(canvas, hit_pt, 8, (0, 220, 255), -1)
    cv2.circle(canvas, bounce_pt, 8, (255, 180, 0), -1)

    cv2.putText(canvas, "HIT", (hit_pt[0] + 8, hit_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BOUNCE", (bounce_pt[0] + 8, bounce_pt[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Step2 trajectory in standard court XY", (margin, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 240, 240), 2, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_png), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Step2: solve initial velocity from Step1 2D + height assumptions")
    parser.add_argument("--video_id", type=str, default="001")
    parser.add_argument("--step1_json", type=str, default="", help="Path to step1_2d_xxx.json")
    parser.add_argument("--z_hit", type=float, default=2.775, help="Hit height (m), default=1.85*1.5")
    parser.add_argument("--z_bounce", type=float, default=0.0, help="Bounce height (m)")
    parser.add_argument("--g", type=float, default=9.81, help="Gravity (m/s^2)")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_plot", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    step1_json = Path(args.step1_json) if args.step1_json else (project_root / "output" / "step1_2d" / args.video_id / "step1_2d.json")
    if not step1_json.is_absolute():
        step1_json = project_root / step1_json

    if not step1_json.exists():
        raise FileNotFoundError(f"step1 json not found: {step1_json}")

    with step1_json.open("r", encoding="utf-8") as f:
        d = json.load(f)

    hit_xy = np.array(d["hit"]["world_xy_m"], dtype=np.float64)
    bounce_xy = np.array(d["bounce"]["world_xy_m"], dtype=np.float64)

    dt = float(d.get("delta_t_sec", d["bounce"]["time_sec"] - d["hit"]["time_sec"]))
    hit_xyz = np.array([hit_xy[0], hit_xy[1], float(args.z_hit)], dtype=np.float64)
    bounce_xyz = np.array([bounce_xy[0], bounce_xy[1], float(args.z_bounce)], dtype=np.float64)

    v0 = solve_initial_velocity(hit_xyz, bounce_xyz, dt, float(args.g))
    speed = float(np.linalg.norm(v0))
    speed_kmh = speed * 3.6
    horizontal_speed = float(math.hypot(v0[0], v0[1]))

    t, xyz = sample_trajectory(hit_xyz, v0, float(args.g), dt, n_points=80)

    out_json_dir = project_root / "output" / "step2_velocity" / args.video_id
    out_traj_dir = project_root / "output" / "step2_trajectory" / args.video_id
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (out_json_dir / "step2_velocity.json")
    out_csv = Path(args.out_csv) if args.out_csv else (out_traj_dir / "step2_trajectory.csv")
    out_plot = Path(args.out_plot) if args.out_plot else (out_traj_dir / "step2_trajectory.png")
    if not out_json.is_absolute():
        out_json = project_root / out_json
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv
    if not out_plot.is_absolute():
        out_plot = project_root / out_plot

    result = {
        "video_id": d.get("video_id", args.video_id),
        "step1_json": str(step1_json),
        "assumption": {
            "z_hit_m": float(args.z_hit),
            "z_bounce_m": float(args.z_bounce),
            "gravity_m_s2": float(args.g),
        },
        "hit_xyz_m": hit_xyz.tolist(),
        "bounce_xyz_m": bounce_xyz.tolist(),
        "delta_t_sec": dt,
        "v0_m_s": {
            "vx": float(v0[0]),
            "vy": float(v0[1]),
            "vz": float(v0[2]),
            "speed": speed,
            "speed_kmh": speed_kmh,
            "horizontal_speed": horizontal_speed,
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "x_m", "y_m", "z_m"])
        for i in range(len(t)):
            writer.writerow([f"{t[i]:.6f}", f"{xyz[i,0]:.6f}", f"{xyz[i,1]:.6f}", f"{xyz[i,2]:.6f}"])

    save_xy_plot_png(
        out_png=out_plot,
        hit_xy=hit_xy,
        bounce_xy=bounce_xy,
        traj_xy=xyz[:, :2],
    )

    print("[STEP2 OK] Initial velocity solved")
    print(f"  step1_json : {step1_json}")
    print(f"  out_json   : {out_json}")
    print(f"  out_csv    : {out_csv}")
    print(f"  out_plot   : {out_plot}")
    print(f"  v0 (m/s)   : vx={v0[0]:.4f}, vy={v0[1]:.4f}, vz={v0[2]:.4f}")
    print(f"  speed      : {speed:.4f} m/s  ({speed_kmh:.2f} km/h)")


if __name__ == "__main__":
    main()
