#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_ball_csv(csv_path: Path) -> List[Tuple[int, int, float, float]]:
    rows: List[Tuple[int, int, float, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Frame", "Visibility", "X", "Y"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"Unexpected CSV format: {csv_path}")
        for row in reader:
            fr = int(float(row["Frame"]))
            vis = int(float(row["Visibility"]))
            x = float(row["X"])
            y = float(row["Y"])
            rows.append((fr, vis, x, y))
    if not rows:
        raise RuntimeError(f"Empty ball CSV: {csv_path}")
    return rows


def load_ball_npy(npy_path: Path) -> List[Tuple[int, int, float, float]]:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected ball npy shape: {arr.shape}")
    rows: List[Tuple[int, int, float, float]] = []
    for i, (x, y) in enumerate(arr, start=1):
        vis = 0 if (x == 0 and y == 0) else 1
        rows.append((i, vis, float(x), float(y)))
    return rows


def find_ball_source(project_root: Path, video_id: str, ball_csv_arg: Optional[str]) -> Tuple[List[Tuple[int, int, float, float]], Path]:
    if ball_csv_arg:
        p = Path(ball_csv_arg)
        if not p.is_absolute():
            p = project_root / p
        return load_ball_csv(p), p

    candidates = [
        project_root / "output" / "tracknetv4_pytorch" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4_pytorch_new" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4" / f"{video_id}_predict_ball.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_ball_csv(p), p

    npy_fallback = project_root / "output" / "ball" / f"{video_id}.npy"
    if npy_fallback.exists():
        return load_ball_npy(npy_fallback), npy_fallback

    raise FileNotFoundError(
        "No ball source found. Checked output/tracknetv4_pytorch, "
        "output/tracknetv4_pytorch_new, output/tracknetv4, and output/ball/{id}.npy"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-frame ball-vs-head height report.")
    parser.add_argument("--video-id", default="002", help="Video id, e.g. 001, 002")
    parser.add_argument("--ball-csv", default=None, help="Optional ball CSV path")
    parser.add_argument(
        "--head-mode",
        default="near",
        choices=["near", "far", "min"],
        help="Which head to use as threshold: near(players[t,1]), far(players[t,0]), or min(two heads)",
    )
    parser.add_argument(
        "--margin-px",
        type=float,
        default=0.0,
        help="Extra margin in pixels. Condition becomes ball_y < (head_y - margin_px)",
    )
    parser.add_argument(
        "--players",
        default=None,
        help="Optional players_only.npy path (default: output/pose_keypoints/players_only.npy)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output txt path (default: output/ball_vs_head_{video_id}.txt)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    players_path = Path(args.players) if args.players else (project_root / "output" / "pose_keypoints" / "players_only.npy")
    if not players_path.is_absolute():
        players_path = project_root / players_path
    if not players_path.exists():
        raise FileNotFoundError(f"players file not found: {players_path}")

    ball_rows, ball_source = find_ball_source(project_root, args.video_id, args.ball_csv)
    players = np.load(players_path).astype(np.float32)

    T = min(len(ball_rows), len(players))
    head0 = players[:T, 0, 0, 1]
    head1 = players[:T, 1, 0, 1]

    out_path = Path(args.out) if args.out else (project_root / "output" / f"ball_vs_head_{args.video_id}.txt")
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    higher_frames: List[int] = []
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Ball vs Head Height Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"video_id: {args.video_id}\n")
        f.write(f"ball_source: {ball_source}\n")
        f.write(f"players_source: {players_path}\n")
        f.write(f"frames_used: {T}\n")
        f.write(f"head_mode: {args.head_mode}\n")
        f.write(f"margin_px: {args.margin_px}\n")
        f.write("rule: ball_higher_than_head = (ball_y < (selected_head_y - margin_px)) in image coordinates\n\n")

        f.write("Frame\tVis\tBall_X\tBall_Y\tHead0_Y\tHead1_Y\tHead_Sel_Y\tBall>Head\n")

        for i in range(T):
            fr, vis, bx, by = ball_rows[i]
            h0 = float(head0[i]) if np.isfinite(head0[i]) else np.nan
            h1 = float(head1[i]) if np.isfinite(head1[i]) else np.nan

            hs = np.array([h0, h1], dtype=np.float32)
            hs = hs[np.isfinite(hs)]
            hmin = float(np.min(hs)) if hs.size > 0 else np.nan

            if args.head_mode == "near":
                hsel = h1
            elif args.head_mode == "far":
                hsel = h0
            else:
                hsel = hmin

            cond = int(
                vis == 1
                and np.isfinite(by)
                and np.isfinite(hsel)
                and (by < (hsel - args.margin_px))
            )
            if cond == 1:
                higher_frames.append(fr)

            f.write(f"{fr}\t{vis}\t{bx:.2f}\t{by:.2f}\t{h0:.2f}\t{h1:.2f}\t{hsel:.2f}\t{cond}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"ball_higher_than_head_count: {len(higher_frames)}\n")
        if higher_frames:
            f.write("frames: " + ",".join(map(str, higher_frames)) + "\n")
        else:
            f.write("frames: (none)\n")

    print(f"saved: {out_path}")
    print(f"ball_higher_than_head_count: {len(higher_frames)}")


if __name__ == "__main__":
    main()
