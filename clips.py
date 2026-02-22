import os
import cv2
import numpy as np
from typing import List, Tuple


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def moving_average_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones((k,), dtype=np.float32) / k
    return np.convolve(xp, ker, mode="valid")


def interpolate_short_gaps_xy(xy: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """Fill gaps of length <= max_gap by linear interpolation. xy: (T,2), (0,0)=missing."""
    xy = xy.astype(np.float32).copy()
    T = xy.shape[0]
    valid = ~((xy[:, 0] == 0) & (xy[:, 1] == 0))
    if valid.sum() < 2:
        return xy

    idx = np.arange(T)
    valid_idx = idx[valid]
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            xa, ya = xy[a]
            xb, yb = xy[b]
            for t in range(1, gap + 1):
                r = t / (gap + 1)
                xy[a + t, 0] = xa + r * (xb - xa)
                xy[a + t, 1] = ya + r * (yb - ya)
    return xy


def dist2(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return float(d[0] * d[0] + d[1] * d[1])


def wrist_over_head(players: np.ndarray, player_id: int, head_idx=0, lw=9, rw=10, margin_px=18) -> np.ndarray:
    """Return bool (T,) wrist above head."""
    k = players[:, player_id]  # (T,J,2)
    head_y = k[:, head_idx, 1]
    lw_y = k[:, lw, 1]
    rw_y = k[:, rw, 1]
    wrist_y = np.minimum(lw_y, rw_y)
    cond = wrist_y < (head_y - margin_px)
    sm = moving_average_1d(cond.astype(np.float32), k=5)
    return sm > 0.6


def find_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    runs = []
    T = len(mask)
    i = 0
    while i < T:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i < T and mask[i]:
            i += 1
        e = i - 1
        if e - s + 1 >= min_len:
            runs.append((s, e))
    return runs


def pick_server_for_run(players: np.ndarray, run: Tuple[int, int], head_idx=0, lw=9, rw=10, margin_px=18) -> int:
    s, e = run
    m0 = wrist_over_head(players[s:e+1], 0, head_idx=head_idx, lw=lw, rw=rw, margin_px=margin_px).sum()
    m1 = wrist_over_head(players[s:e+1], 1, head_idx=head_idx, lw=lw, rw=rw, margin_px=margin_px).sum()
    return 0 if m0 >= m1 else 1


def first_ball_near_server(ball_xy: np.ndarray, players: np.ndarray, player_id: int, t0: int, t1: int,
                           lw=9, rw=10, dist_thresh_px=220) -> int:
    thr2 = dist_thresh_px * dist_thresh_px
    for t in range(t0, t1 + 1):
        bx, by = ball_xy[t]
        if bx == 0 and by == 0:
            continue
        b = np.array([bx, by], dtype=np.float32)
        w1 = players[t, player_id, lw]
        w2 = players[t, player_id, rw]
        if dist2(b, w1) <= thr2 or dist2(b, w2) <= thr2:
            return t
    return -1


def estimate_hit_frame(ball_xy_f: np.ndarray, players: np.ndarray, player_id: int, t_start: int,
                       max_search: int, lw=9, rw=10,
                       near_thresh_px=220, far_thresh_px=420, speed_thresh_px=20.0) -> int:
    """
    More conservative hit:
      - find first time ball near wrist
      - then first time ball becomes far AND speed large
    """
    near2 = near_thresh_px * near_thresh_px
    far2 = far_thresh_px * far_thresh_px
    T = len(ball_xy_f)
    t_end = min(T - 2, t_start + max_search)

    t_near = -1
    for t in range(t_start, t_end + 1):
        bx, by = ball_xy_f[t]
        if bx == 0 and by == 0:
            continue
        b = np.array([bx, by], dtype=np.float32)
        w1 = players[t, player_id, lw]
        w2 = players[t, player_id, rw]
        if dist2(b, w1) <= near2 or dist2(b, w2) <= near2:
            t_near = t
            break
    if t_near == -1:
        return -1

    for t in range(t_near + 1, t_end + 1):
        bx1, by1 = ball_xy_f[t]
        bx2, by2 = ball_xy_f[t + 1]
        if (bx1 == 0 and by1 == 0) or (bx2 == 0 and by2 == 0):
            continue

        b = np.array([bx1, by1], dtype=np.float32)
        w1 = players[t, player_id, lw]
        w2 = players[t, player_id, rw]
        d2min = min(dist2(b, w1), dist2(b, w2))

        vx = float(bx2 - bx1)
        vy = float(by2 - by1)
        speed = (vx * vx + vy * vy) ** 0.5

        if d2min >= far2 and speed >= speed_thresh_px:
            return t

    return t_near


def homography_from_line20(court20: np.ndarray) -> np.ndarray:
    """
    Use 4 corners:
      idx2 near-left  -> (0,0)
      idx9 near-right -> (8.23,0)
      idx8 far-right  -> (8.23,23.77)
      idx6 far-left   -> (0,23.77)
    Returns H mapping image(x,y)->world(X,Y)
    """
    p2 = court20[2]
    p9 = court20[9]
    p8 = court20[8]
    p6 = court20[6]

    src = np.array([p2, p9, p8, p6], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [8.23, 0.0], [8.23, 23.77], [0.0, 23.77]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def img_to_world(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """xy: (N,2) float -> (N,2) world"""
    pts = xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def find_first_bounce_in_service_box(ball_xy_f: np.ndarray, line: np.ndarray, t_hit: int, server_id: int,
                                     fps: float, max_search_sec: float = 3.0,
                                     smooth_k: int = 7) -> int:
    """
    Bounce = first local max of y AFTER hit, but ONLY when ball is inside target service box in WORLD coords.
    This removes later rally / out-of-box detections.
    """
    T = len(ball_xy_f)
    t_end = min(T - 2, t_hit + int(max_search_sec * fps))

    # service box world Y ranges
    net_y = 23.77 / 2.0          # 11.885
    sline_near = net_y - 6.40    # 5.485
    sline_far = net_y + 6.40     # 18.285

    # server_id: 1=bottom(near) serve -> bounce in far service box
    #           0=top(far) serve    -> bounce in near service box
    if server_id == 1:
        y_min, y_max = net_y, sline_far
    else:
        y_min, y_max = sline_near, net_y

    # Build H for each frame (line is per-frame)
    # Use earliest stable frame near t_hit for H
    court20 = line[min(t_hit, line.shape[0] - 1)].reshape(20, 2)
    H = homography_from_line20(court20)

    # Prepare ball segment
    seg = ball_xy_f[t_hit:t_end + 1].copy()
    valid = ~((seg[:, 0] == 0) & (seg[:, 1] == 0))
    if valid.sum() < 8:
        return -1

    # Smooth pixel y for local max check
    y = seg[:, 1].astype(np.float32)
    y_s = moving_average_1d(y, k=smooth_k)
    dy = np.diff(y_s)

    # World coordinates for valid points
    seg_world = np.full_like(seg, np.nan, dtype=np.float32)
    vidx = np.where(valid)[0]
    wpts = img_to_world(H, seg[vidx])
    seg_world[vidx] = wpts

    # Find earliest local max that is inside service box
    for i in range(1, len(dy)):
        # local max at i when dy[i-1]>0 and dy[i]<0
        if dy[i - 1] > 0 and dy[i] < 0:
            wy = seg_world[i, 1]
            wx = seg_world[i, 0]
            if np.isnan(wx) or np.isnan(wy):
                continue
            # inside singles court and target service box
            if 0.0 <= wx <= 8.23 and (y_min <= wy <= y_max):
                return t_hit + i

    return -1


def cut_segments(video_path: str, segments: List[Tuple[int, int]], out_dir: str):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for i, (s, e) in enumerate(segments, 1):
        s = max(0, min(int(s), total - 1))
        e = max(0, min(int(e), total - 1))
        if e <= s:
            continue

        out_path = os.path.join(out_dir, f"{i:03d}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        f = s
        while f <= e:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            f += 1

        writer.release()
        print(f"[OK] saved {out_path}  frames={e - s + 1}  range=[{s},{e}]")

    cap.release()


def main():
    video_path = "/root/tennis-analysis/videos/001.mp4"
    players_path = "/root/tennis-analysis/output/pose_dump/players_only.npy"
    ball_path = "/root/tennis-analysis/output/ball/001.npy"
    line_path = "/root/tennis-analysis/output/line/001.npy"
    out_dir = "/root/tennis-analysis/output/cuts_serve_clean"

    players = np.load(players_path)  # (T,2,17,2)
    ball = np.load(ball_path)        # (T,2) int64, [0,0]=missing
    line = np.load(line_path)        # (T,40) -> (20,2)

    T = min(players.shape[0], ball.shape[0], line.shape[0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    players = players[:T]
    ball = ball[:T]
    line = line[:T]

    # ---- tunable params (重点你先动这几个) ----
    HEAD = 0
    LW, RW = 9, 10

    ARM_MARGIN_PX = 20               # 更严格一点，减少误触发
    MIN_ARM_HOLD = int(0.20 * fps)   # 连续0.2秒举手才算
    BALL_NEAR_WRIST_PX = 220         # 更严格：球必须靠近手
    HIT_SEARCH_SEC = 2.0
    BOUNCE_SEARCH_SEC = 3.0

    # 让剪出来更“干净”
    PRE_SEC = 0.20                   # start 前只留 0.2s
    POST_SEC = 0.25                  # bounce 后只留 0.25s

    # 球插值：稍微放宽一点，避免落地附近漏检导致 bounce 推迟
    ball_f = interpolate_short_gaps_xy(ball, max_gap=8)

    arm0 = wrist_over_head(players, 0, head_idx=HEAD, lw=LW, rw=RW, margin_px=ARM_MARGIN_PX)
    arm1 = wrist_over_head(players, 1, head_idx=HEAD, lw=LW, rw=RW, margin_px=ARM_MARGIN_PX)
    arm_any = arm0 | arm1
    runs = find_runs(arm_any, min_len=MIN_ARM_HOLD)

    print(f"[INFO] arm-up runs: {len(runs)}")

    segments: List[Tuple[int, int]] = []
    cooldown = 0
    cooldown_frames = int(2.0 * fps)

    for (rs, re) in runs:
        if cooldown > 0:
            cooldown -= 1
            continue

        server_id = pick_server_for_run(players, (rs, re), head_idx=HEAD, lw=LW, rw=RW, margin_px=ARM_MARGIN_PX)

        t_ball_near = first_ball_near_server(
            ball_xy=ball,
            players=players,
            player_id=server_id,
            t0=rs,
            t1=re,
            lw=LW,
            rw=RW,
            dist_thresh_px=BALL_NEAR_WRIST_PX,
        )
        if t_ball_near == -1:
            continue

        # hit
        t_hit = estimate_hit_frame(
            ball_xy_f=ball_f,
            players=players,
            player_id=server_id,
            t_start=t_ball_near,
            max_search=int(HIT_SEARCH_SEC * fps),
            lw=LW,
            rw=RW,
            near_thresh_px=BALL_NEAR_WRIST_PX,
            far_thresh_px=420,
            speed_thresh_px=20.0,
        )
        if t_hit == -1:
            continue

        # bounce (service box constrained)
        t_bounce = find_first_bounce_in_service_box(
            ball_xy_f=ball_f,
            line=line,
            t_hit=t_hit,
            server_id=server_id,
            fps=fps,
            max_search_sec=BOUNCE_SEARCH_SEC,
            smooth_k=7,
        )
        if t_bounce == -1:
            # fallback: old way -> just stop at hit+window
            t_bounce = min(T - 1, t_hit + int(BOUNCE_SEARCH_SEC * fps))

        # ✅ 更干净的 start：以 arm run 起点为准，而不是 ball_near
        start = max(0, rs - int(PRE_SEC * fps))
        end = min(T - 1, t_bounce + int(POST_SEC * fps))

        segments.append((start, end))
        cooldown = cooldown_frames

        print(
            f"[SEG] server={'TOP' if server_id==0 else 'BOTTOM'} "
            f"arm_run=[{rs},{re}] ball_near={t_ball_near} hit={t_hit} bounce={t_bounce} "
            f"cut=[{start},{end}] sec=[{start/fps:.2f},{end/fps:.2f}]"
        )

    if not segments:
        print("[WARN] no segments detected.")
        return

    # merge overlaps
    segments.sort()
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s <= pe + int(0.3 * fps):
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    segments = merged

    print(f"[INFO] final segments: {len(segments)}")
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "segments.npy"), np.array(segments, dtype=np.int32))
    cut_segments(video_path, segments, out_dir)


if __name__ == "__main__":
    main()