import argparse
import math

import numpy as np


def _line_angle_deg(line4: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line4)
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _line_len(line4: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line4)
    return math.hypot(x2 - x1, y2 - y1)


def main():
    ap = argparse.ArgumentParser(description="Inspect court line indices (first frame)")
    ap.add_argument("--line", required=True, help="Court lines .npy")
    args = ap.parse_args()

    lines = np.load(args.line, allow_pickle=True)
    if len(lines) < 1:
        raise RuntimeError("Empty line npy")
    l0 = np.asarray(lines[0], dtype=np.float32).reshape(-1, 4)
    if l0.shape[0] != 10:
        raise RuntimeError(f"Expected 10 lines, got {l0.shape}")

    print("idx  len(px)  angle(deg)   mid(x,y)")
    for i, l in enumerate(l0):
        ln = _line_len(l)
        ang = _line_angle_deg(l)
        mx = 0.5 * (float(l[0]) + float(l[2]))
        my = 0.5 * (float(l[1]) + float(l[3]))
        print(f"{i:>3d}  {ln:>7.1f}  {ang:>9.1f}   ({mx:>6.1f},{my:>6.1f})")


if __name__ == "__main__":
    main()
