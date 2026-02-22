# /root/tennis-analysis/pose_filter_players_by_line.py
import os
import glob
import numpy as np
import cv2

from rtmlib.rtmlib.visualization import draw_skeleton


# ---------- geometry utils ----------
def line_abc_from_2pts(p1, p2):
    """Line through p1,p2 in ax+by+c=0 form."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = (a * a + b * b) ** 0.5
    if norm < 1e-6:
        return 0.0, 0.0, 1e9
    return a, b, c


def point_line_distance(p, line_abc):
    x, y = float(p[0]), float(p[1])
    a, b, c = line_abc
    return abs(a * x + b * y + c) / ((a * a + b * b) ** 0.5 + 1e-9)


def safe_foot_midpoint(kpt):
    """
    kpt: (J,2). assume last two joints are feet.
    return None if invalid.
    """
    if kpt is None:
        return None
    kpt = np.asarray(kpt)
    if kpt.ndim != 2 or kpt.shape[1] != 2 or kpt.shape[0] < 2:
        return None
    a = kpt[-2]
    b = kpt[-1]
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None
    return (a + b) / 2.0


def x_on_segment_at_y(p1, p2, y):
    """
    For a segment p1->p2, get x coordinate at a given y by linear interpolation.
    Works well for court side lines under perspective.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2.0
    t = (float(y) - y1) / (y2 - y1)
    return x1 + (x2 - x1) * t


def select_players_by_baselines(kpts_people, court20, x_margin=30, max_d_px=None):
    """
    kpts_people: (P,J,2) float
    court20: (20,2) float
    Returns (top_idx, bottom_idx) or (None,None)

    Key idea:
      1) Use perspective corridor gate: x must be between left/right sideline at that y (plus margin)
      2) Then choose closest to far/near baseline within top/bottom half.
    """
    if kpts_people is None or len(kpts_people) == 0:
        return None, None

    # baseline endpoints (your confirmed indices)
    p2 = court20[2]   # near left
    p9 = court20[9]   # near right
    p6 = court20[6]   # far left
    p8 = court20[8]   # far right

    near_line = line_abc_from_2pts(p2, p9)
    far_line = line_abc_from_2pts(p6, p8)

    # rough net/middle separator: average y of four corners
    mid_y = float((p2[1] + p9[1] + p6[1] + p8[1]) / 4.0)

    # perspective side lines (left: p2->p6, right: p9->p8)
    left_a, left_b = p2, p6
    right_a, right_b = p9, p8

    top_best = (1e18, None)       # (distance, idx)
    bottom_best = (1e18, None)

    for i, kpt in enumerate(kpts_people):
        fm = safe_foot_midpoint(kpt)
        if fm is None:
            continue

        x = float(fm[0])
        y = float(fm[1])

        # Perspective x-range at this y
        x_left = x_on_segment_at_y(left_a, left_b, y)
        x_right = x_on_segment_at_y(right_a, right_b, y)
        if x_left > x_right:
            x_left, x_right = x_right, x_left

        # Gate by corridor (this is the key to remove umpire / side persons)
        if not (x_left - x_margin <= x <= x_right + x_margin):
            continue

        d_near = point_line_distance(fm, near_line)
        d_far = point_line_distance(fm, far_line)

        # optional distance gate to remove far-away people
        if max_d_px is not None:
            if y >= mid_y and d_near > max_d_px:
                continue
            if y < mid_y and d_far > max_d_px:
                continue

        if y < mid_y:
            # top half: choose closest to far baseline
            if d_far < top_best[0]:
                top_best = (d_far, i)
        else:
            # bottom half: choose closest to near baseline
            if d_near < bottom_best[0]:
                bottom_best = (d_near, i)

    return top_best[1], bottom_best[1]


# ---------- IO helpers ----------
def find_pose_npz(pose_dump_dir):
    # Prefer common names
    candidates = [
        os.path.join(pose_dump_dir, "pose_dump_all_frames.npz"),
        os.path.join(pose_dump_dir, "pose_dump.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fallback: first npz in dir
    lst = sorted(glob.glob(os.path.join(pose_dump_dir, "*.npz")))
    return lst[0] if lst else None


def load_pose_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "keypoints" not in data:
        raise KeyError(f"'keypoints' not found in {npz_path}. keys={list(data.keys())}")
    kpts = data["keypoints"]  # object array: len=frames, each (P,J,2)
    scores = data["scores"] if "scores" in data else None
    meta = {k: data[k] for k in data.files if k not in ("keypoints", "scores")}
    return kpts, scores, meta


def main():
    pose_dump_dir = "/root/tennis-analysis/output/pose_dump"
    line_path = "/root/tennis-analysis/output/line/001.npy"
    out_path = "/root/tennis-analysis/output/pose_dump/players_only.npy"

    # ----- visualization options -----
    SAVE_VIS = True
    VIDEO_PATH = "/root/tennis-analysis/videos/001.mp4"
    VIS_OUT_PATH = "/root/tennis-analysis/output/pose_dump/players_only_vis.mp4"
    KPT_THR = 0.3  # drawing threshold (lower -> draw more)

    # ----- selection tuning -----
    # smaller -> stricter (less likely to pick umpire), larger -> safer for players stepping out
    X_MARGIN = 30
    # Optional: set e.g. 160 to enforce "must be near baseline" more strictly; keep None first
    MAX_D_PX = None

    pose_npz = find_pose_npz(pose_dump_dir)
    if pose_npz is None:
        raise FileNotFoundError(f"No .npz pose dump found in: {pose_dump_dir}")

    if not os.path.isfile(line_path):
        raise FileNotFoundError(f"Line file not found: {line_path}")

    print(f"[INFO] pose_npz: {pose_npz}")
    print(f"[INFO] line_npy: {line_path}")

    # load
    kpts_obj, scores_obj, _ = load_pose_npz(pose_npz)
    line = np.load(line_path)  # (T,40) float32
    assert line.ndim == 2 and line.shape[1] == 40, f"Unexpected line shape: {line.shape}"

    T_line = line.shape[0]
    T_pose = len(kpts_obj)
    T = min(T_pose, T_line)
    if T_pose != T_line:
        print(f"[WARN] frame mismatch: pose={T_pose}, line={T_line}, use T={T}")

    # filter
    players_frames = []          # each: (2,J,2) float32, order: [top, bottom]
    players_scores_frames = []   # each: (2,J) float32, order: [top, bottom]
    missing_top = 0
    missing_bottom = 0

    for t in range(T):
        court20 = line[t].reshape(20, 2)
        people = kpts_obj[t]  # (P,J,2) or empty

        if people is None or len(people) == 0:
            players_frames.append(np.full((2, 0, 2), np.nan, dtype=np.float32))
            players_scores_frames.append(np.full((2, 0), np.nan, dtype=np.float32))
            missing_top += 1
            missing_bottom += 1
            continue

        people = np.asarray(people, dtype=np.float32)
        P, J, _ = people.shape

        top_i, bottom_i = select_players_by_baselines(
            people,
            court20,
            x_margin=X_MARGIN,
            max_d_px=MAX_D_PX,
        )

        out = np.full((2, J, 2), np.nan, dtype=np.float32)
        out_s = np.full((2, J), np.nan, dtype=np.float32)

        frame_scores = None
        if scores_obj is not None:
            frame_scores = scores_obj[t]
            if frame_scores is not None and len(frame_scores) > 0:
                frame_scores = np.asarray(frame_scores, dtype=np.float32)  # (P,J)

        if top_i is not None:
            out[0] = people[top_i]
            if frame_scores is not None and top_i < frame_scores.shape[0]:
                out_s[0] = frame_scores[top_i]
        else:
            missing_top += 1

        if bottom_i is not None:
            out[1] = people[bottom_i]
            if frame_scores is not None and bottom_i < frame_scores.shape[0]:
                out_s[1] = frame_scores[bottom_i]
        else:
            missing_bottom += 1

        players_frames.append(out)
        players_scores_frames.append(out_s)

    players = np.stack(players_frames, axis=0)               # (T,2,J,2)
    players_scores = np.stack(players_scores_frames, axis=0) # (T,2,J)

    # save
    np.save(out_path, players)
    print(f"[OK] saved players_only.npy -> {out_path}")

    scores_out_path = out_path.replace(".npy", "_scores.npy")
    np.save(scores_out_path, players_scores)
    print(f"[OK] saved players_only_scores.npy -> {scores_out_path}")

    print(f"[INFO] shape={players.shape} (frames, 2, joints, 2)")
    print(f"[INFO] missing_top={missing_top}/{T}, missing_bottom={missing_bottom}/{T}")
    print("\nOrder: players[t,0]=TOP (far side), players[t,1]=BOTTOM (near side)")

    # ----- make visualization video -----
    if SAVE_VIS:
        if not os.path.isfile(VIDEO_PATH):
            raise FileNotFoundError(f"Video not found for visualization: {VIDEO_PATH}")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIS_OUT_PATH, fourcc, fps, (w, h))

        t2 = players.shape[0]
        t = 0
        while True:
            ret, frame = cap.read()
            if not ret or t >= t2:
                break

            # keypoints: (2, J, 2)
            kpts_arr = np.stack([players[t, 0], players[t, 1]], axis=0).astype(np.float32)

            # scores: (2, J)
            s0 = players_scores[t, 0]
            s1 = players_scores[t, 1]

            if s0 is None or np.all(np.isnan(s0)):
                s0 = np.ones((players.shape[2],), dtype=np.float32)
            else:
                s0 = s0.astype(np.float32)

            if s1 is None or np.all(np.isnan(s1)):
                s1 = np.ones((players.shape[2],), dtype=np.float32)
            else:
                s1 = s1.astype(np.float32)

            scores_arr = np.stack([s0, s1], axis=0)

            frame = draw_skeleton(frame, kpts_arr, scores_arr, kpt_thr=KPT_THR)
            writer.write(frame)
            t += 1

        cap.release()
        writer.release()
        print(f"[OK] saved visualization video -> {VIS_OUT_PATH}")


if __name__ == "__main__":
    main()