#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def draw_label_near_point(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font,
    font_scale: float,
    thickness: int,
) -> None:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    tx = x + 8
    ty = y - 8

    if tx + tw + 4 > w:
        tx = max(2, x - tw - 8)
    if ty - th - 4 < 0:
        ty = min(h - 2, y + th + 8)

    x1 = max(0, tx - 2)
    y1 = max(0, ty - th - 2)
    x2 = min(w - 1, tx + tw + 2)
    y2 = min(h - 1, ty + 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def load_ball_from_csv(csv_path: Path) -> Dict[int, Tuple[int, float, float]]:
    data: Dict[int, Tuple[int, float, float]] = {}
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
            data[fr] = (vis, x, y)
    return data


def load_ball_from_npy(npy_path: Path) -> Dict[int, Tuple[int, float, float]]:
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected ball npy shape: {arr.shape}")
    data: Dict[int, Tuple[int, float, float]] = {}
    for i, (x, y) in enumerate(arr, start=1):
        vis = 0 if (x == 0 and y == 0) else 1
        data[i] = (vis, float(x), float(y))
    return data


def find_ball_source(project_root: Path, video_id: str, ball_csv_arg: str) -> Tuple[Dict[int, Tuple[int, float, float]], Path]:
    if ball_csv_arg:
        p = Path(ball_csv_arg)
        if not p.is_absolute():
            p = project_root / p
        return load_ball_from_csv(p), p

    candidates = [
        project_root / "output" / "tracknetv4_pytorch" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4_pytorch_new" / f"{video_id}_predict_ball.csv",
        project_root / "output" / "tracknetv4" / f"{video_id}_predict_ball.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_ball_from_csv(p), p

    npy_fallback = project_root / "output" / "ball" / f"{video_id}.npy"
    if npy_fallback.exists():
        return load_ball_from_npy(npy_fallback), npy_fallback

    raise FileNotFoundError("No ball source found in output/tracknetv4* or output/ball/{id}.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create debug overlay video: red ball dot + per-frame coordinates text.")
    parser.add_argument("--video-id", default="002", help="Video id, e.g. 001, 002")
    parser.add_argument("--ball-csv", default="", help="Optional ball CSV path")
    parser.add_argument("--video", default="", help="Optional input video path")
    parser.add_argument("--players", default="", help="Optional players_only.npy path")
    parser.add_argument("--out", default="", help="Optional output video path")
    parser.add_argument("--dot-radius", type=int, default=4, help="Ball dot radius")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    video_path = Path(args.video) if args.video else (project_root / "videos" / f"{args.video_id}.mp4")
    if not video_path.is_absolute():
        video_path = project_root / video_path

    players_path = Path(args.players) if args.players else (project_root / "output" / "pose_keypoints" / "players_only.npy")
    if not players_path.is_absolute():
        players_path = project_root / players_path

    out_path = Path(args.out) if args.out else (project_root / "output" / f"{args.video_id}_ball_head_overlay_debug.mp4")
    if not out_path.is_absolute():
        out_path = project_root / out_path

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not players_path.exists():
        raise FileNotFoundError(f"players_only.npy not found: {players_path}")

    ball, ball_source = find_ball_source(project_root, args.video_id, args.ball_csv)
    players = np.load(players_path).astype(np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    line_h = 18

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx - 1 < len(players):
            h0x, h0y = players[frame_idx - 1, 0, 0, 0], players[frame_idx - 1, 0, 0, 1]
            h1x, h1y = players[frame_idx - 1, 1, 0, 0], players[frame_idx - 1, 1, 0, 1]
        else:
            h0x = h0y = h1x = h1y = np.nan

        vis, bx, by = ball.get(frame_idx, (0, -1.0, -1.0))

        if vis == 1 and bx >= 0 and by >= 0:
            bxi, byi = int(round(bx)), int(round(by))
            cv2.circle(frame, (bxi, byi), args.dot_radius, (0, 0, 255), -1)
            draw_label_near_point(
                frame,
                f"ball({bx:.1f},{by:.1f})",
                bxi,
                byi,
                (255, 255, 255),
                font,
                font_scale,
                thickness,
            )

        if np.isfinite(h0x) and np.isfinite(h0y):
            h0xi, h0yi = int(round(h0x)), int(round(h0y))
            cv2.circle(frame, (h0xi, h0yi), 3, (255, 255, 0), -1)
            draw_label_near_point(
                frame,
                f"h0({h0x:.1f},{h0y:.1f})",
                h0xi,
                h0yi,
                (255, 255, 0),
                font,
                font_scale,
                thickness,
            )
        if np.isfinite(h1x) and np.isfinite(h1y):
            h1xi, h1yi = int(round(h1x)), int(round(h1y))
            cv2.circle(frame, (h1xi, h1yi), 3, (0, 255, 255), -1)
            draw_label_near_point(
                frame,
                f"h1({h1x:.1f},{h1y:.1f})",
                h1xi,
                h1yi,
                (0, 255, 255),
                font,
                font_scale,
                thickness,
            )

        # Keep frame id in a small corner tag.
        cv2.rectangle(frame, (8, 6), (170, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total}", (12, 24), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"video_source: {video_path}")
    print(f"ball_source: {ball_source}")
    print(f"players_source: {players_path}")
    print(f"saved_video: {out_path}")
    print(f"frames_written: {frame_idx - 1}")


if __name__ == "__main__":
    main()
