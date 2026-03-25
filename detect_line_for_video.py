import argparse
from pathlib import Path

import cv2
import numpy as np

from court_detector import CourtDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Detect court line and save output/line/{video_id}.npy")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--out_npy", type=str, required=True)
    parser.add_argument("--candidate_frames", type=int, nargs="*", default=[0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140])
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    out_npy = Path(args.out_npy)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates = [f for f in args.candidate_frames if 0 <= f < max(total_frames, 1)]
    if not candidates:
        candidates = [0]

    detector = CourtDetector(verbose=0)
    line40 = None
    used_frame = None

    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        try:
            line = detector.detect(frame, verbose=0)
        except Exception:
            line = None

        if line is None:
            continue

        arr = np.asarray(line, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 40:
            continue
        if not np.isfinite(arr).all():
            continue

        line40 = arr
        used_frame = idx
        break

    cap.release()

    if line40 is None:
        raise RuntimeError("Court detection failed on all candidate frames")

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out = np.tile(line40.reshape(1, 40), (max(total_frames, 1), 1)).astype(np.float32)
    np.save(str(out_npy), out)

    print(f"[OK] detected frame: {used_frame}")
    print(f"[OK] saved: {out_npy}")
    print(f"[OK] shape: {out.shape}")


if __name__ == "__main__":
    main()
