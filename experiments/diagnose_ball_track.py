import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_ball(ball_path: str):
    b = np.load(ball_path)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"ball npy must be (T,2); got {b.shape}")
    mask = (b[:, 0] != 0) & (b[:, 1] != 0) & np.isfinite(b[:, 0]) & np.isfinite(b[:, 1])
    return b, mask


def interpolate_2d(points: np.ndarray, mask: np.ndarray):
    T = points.shape[0]
    out = points.copy()
    idx = np.arange(T)
    for d in range(2):
        valid = mask & np.isfinite(points[:, d])
        if valid.sum() < 5:
            raise RuntimeError("Too few valid ball detections to interpolate.")
        out[:, d] = np.interp(idx, idx[valid], points[valid, d])
    return out


def smooth_2d(uv: np.ndarray, ksize: int = 7):
    k = int(ksize)
    if k < 3:
        return uv
    if k % 2 == 0:
        k += 1
    pad = k // 2
    out = uv.copy().astype(np.float64)
    for d in range(2):
        x = out[:, d]
        xp = np.pad(x, (pad, pad), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(xp, k)
        out[:, d] = np.median(win, axis=1)
    return out


def pick_flight_segment_from_accel(
    uv: np.ndarray,
    fps: float,
    min_len_s: float = 0.25,
    max_len_s: float = 0.8,
):
    T = uv.shape[0]
    if T < 8:
        return 0, T

    v = (uv[1:] - uv[:-1]) * fps
    a = (v[1:] - v[:-1]) * fps
    acc = np.linalg.norm(a, axis=1)
    if acc.size < 10:
        return 0, T

    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med)) + 1e-6)
    thr = med + 8.0 * 1.4826 * mad
    peaks = np.where(acc > thr)[0]
    if peaks.size == 0:
        spd = np.linalg.norm(v, axis=1)
        peak = int(np.argmax(spd))
        start = peak
        end = min(T, start + int(round(max_len_s * fps)))
        return start, max(start + 2, end)

    peaks = peaks[np.argsort(-acc[peaks])]
    serve_peak = int(peaks[0])
    start = max(0, serve_peak - 1)

    min_len = int(round(min_len_s * fps))
    max_len = int(round(max_len_s * fps))
    end_min = min(T, start + max(min_len, 2))
    end_max = min(T, start + max_len)

    end = None
    for p in peaks[1:]:
        f = int(p) + 1
        if end_min <= f <= end_max:
            end = f
            break
    if end is None:
        end = end_max
    if end <= start + 1:
        end = min(T, start + 2)
    return start, end


def main():
    ap = argparse.ArgumentParser(description="Diagnose 2D ball track quality")
    ap.add_argument("--ball", required=True, help="Ball detections .npy (T,2)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out_dir", default="output/diagnostics")
    ap.add_argument("--smooth", type=int, default=7, help="Median smoothing kernel")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ball_raw, mask = load_ball(args.ball)
    uv = interpolate_2d(ball_raw, mask)
    uv_s = smooth_2d(uv, ksize=args.smooth)

    v = (uv_s[1:] - uv_s[:-1]) * args.fps
    a = (v[1:] - v[:-1]) * args.fps
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(a, axis=1)

    seg_start, seg_end = pick_flight_segment_from_accel(uv_s, args.fps)

    # plot trajectory
    fig = plt.figure(figsize=(10, 4))
    plt.plot(uv_s[:, 0], uv_s[:, 1], color="#2b6cb0", linewidth=1.5)
    plt.scatter(uv_s[seg_start:seg_end, 0], uv_s[seg_start:seg_end, 1], s=10, color="#d53f8c")
    plt.gca().invert_yaxis()
    plt.title("Ball Track (smoothed) + selected segment")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "track_segment.png"), dpi=160)
    plt.close(fig)

    # speed/acc plots
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(speed, color="#2f855a")
    ax1.axvline(seg_start, color="#d53f8c", linestyle="--")
    ax1.axvline(seg_end, color="#d53f8c", linestyle="--")
    ax1.set_title("Speed (px/s)")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(acc, color="#c53030")
    ax2.axvline(seg_start, color="#d53f8c", linestyle="--")
    ax2.axvline(seg_end, color="#d53f8c", linestyle="--")
    ax2.set_title("Acceleration (px/s^2)")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "speed_acc.png"), dpi=160)
    plt.close(fig)

    print("=== Ball track diagnostics ===")
    print(f"frames={len(uv)}  valid_raw={int(mask.sum())}")
    print(f"segment [{seg_start}, {seg_end}) len={seg_end - seg_start}")
    print(f"speed px/s: median={np.median(speed):.1f} p95={np.quantile(speed, 0.95):.1f} max={np.max(speed):.1f}")
    print(f"acc px/s^2: median={np.median(acc):.1f} p95={np.quantile(acc, 0.95):.1f} max={np.max(acc):.1f}")
    print(f"saved: {os.path.join(args.out_dir, 'track_segment.png')}")
    print(f"saved: {os.path.join(args.out_dir, 'speed_acc.png')}")


if __name__ == "__main__":
    main()
