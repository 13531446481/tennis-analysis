import argparse
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "scipy is required for 2D->3D reconstruction (least_squares). "
        "Install it (e.g. pip install scipy) and retry."
    ) from e


# ---------------------------
# Court geometry
#
# Note: the bundled CourtDetector returns both outer (doubles) and inner (singles)
# lines. If you use the outer boundary lines for corners, court width should be
# doubles (10.97m). The default here is doubles to match indices 3/4.
# ---------------------------
COURT_W_DOUBLES = 10.97  # meters
COURT_W_SINGLES = 8.23   # meters
COURT_L = 23.77          # meters
NET_Y = COURT_L / 2.0
SERVICE_LINE_FROM_NET = 6.40  # meters
SERVICE_Y_NEAR = NET_Y - SERVICE_LINE_FROM_NET
SERVICE_Y_FAR = NET_Y + SERVICE_LINE_FROM_NET
G = 9.81


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    W: int
    H: int


def read_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return float(fps), W, H


def reshape_lines(lines_1d: np.ndarray) -> np.ndarray:
    """(40,) -> (10,4) float32."""
    arr = np.asarray(lines_1d, dtype=np.float32).reshape(-1, 4)
    if arr.shape != (10, 4):
        raise ValueError(f"Expected 40-length lines -> (10,4); got {arr.shape}")
    return arr


def line_intersection(p1, p2, p3, p4, eps=1e-6):
    """Intersect infinite lines p1-p2 and p3-p4. Return (x,y) or None."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float32)


def _line_angle_deg(line4: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line4)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return ang


def _line_len(line4: np.ndarray) -> float:
    x1, y1, x2, y2 = map(float, line4)
    return math.hypot(x2 - x1, y2 - y1)


def pick_court_boundary_auto(lines10: np.ndarray):
    """
    Try to pick: far baseline, near baseline, left boundary, right boundary.

    Heuristic:
    - baselines: the two longest ~horizontal lines
    - boundaries: the two longest ~vertical lines
    """
    lines = np.asarray(lines10, dtype=np.float32)
    angles = np.array([_line_angle_deg(l) for l in lines])
    lens = np.array([_line_len(l) for l in lines])

    # horizontal if |angle| close to 0/180
    horiz_score = np.minimum(np.abs(angles), np.abs(np.abs(angles) - 180.0))

    # vertical-ish lines in this detector can be slanted (~60/120 deg), not only ~90.
    # Use distance to the nearest of {60, 90, 120}.
    cand = np.array([60.0, 90.0, 120.0], dtype=np.float32)[None, :]
    vert_score = np.min(np.abs(np.abs(angles[:, None]) - cand), axis=1)

    horiz_idx = np.argsort(horiz_score + 1e-3 / (lens + 1e-6))
    vert_idx = np.argsort(vert_score + 1e-3 / (lens + 1e-6))

    # take top candidates, then choose the longest two
    h_cand = horiz_idx[:6]
    v_cand = vert_idx[:6]
    h_best = h_cand[np.argsort(-lens[h_cand])][:2]
    v_best = v_cand[np.argsort(-lens[v_cand])][:2]

    h1, h2 = lines[h_best[0]], lines[h_best[1]]
    v1, v2 = lines[v_best[0]], lines[v_best[1]]

    # decide far vs near by y position (smaller y is farther in broadcast views)
    h1y = 0.5 * (h1[1] + h1[3])
    h2y = 0.5 * (h2[1] + h2[3])
    far, near = (h1, h2) if h1y < h2y else (h2, h1)

    # decide left vs right by x position
    v1x = 0.5 * (v1[0] + v1[2])
    v2x = 0.5 * (v2[0] + v2[2])
    left, right = (v1, v2) if v1x < v2x else (v2, v1)

    return far, near, left, right


def pick_court_boundary_by_extremes(lines10: np.ndarray):
    """
    More robust fallback:
    - baselines: horizontal-ish lines, pick top-most and bottom-most by y
    - boundaries: vertical-ish (incl slanted) lines, pick left-most and right-most by x
    """
    lines = np.asarray(lines10, dtype=np.float32)
    angles = np.array([_line_angle_deg(l) for l in lines])
    lens = np.array([_line_len(l) for l in lines])

    horiz_score = np.minimum(np.abs(angles), np.abs(np.abs(angles) - 180.0))
    is_h = horiz_score < 20.0
    if np.sum(is_h) < 2:
        # use best-scored horizontals
        h_idx = np.argsort(horiz_score + 1e-3 / (lens + 1e-6))[:2]
    else:
        h_idx = np.where(is_h)[0]

    h_lines = lines[h_idx]
    h_mid_y = 0.5 * (h_lines[:, 1] + h_lines[:, 3])
    far = h_lines[int(np.argmin(h_mid_y))]
    near = h_lines[int(np.argmax(h_mid_y))]

    cand = np.array([60.0, 90.0, 120.0], dtype=np.float32)[None, :]
    vert_score = np.min(np.abs(np.abs(angles[:, None]) - cand), axis=1)
    is_v = vert_score < 20.0
    if np.sum(is_v) < 2:
        v_idx = np.argsort(vert_score + 1e-3 / (lens + 1e-6))[:2]
    else:
        v_idx = np.where(is_v)[0]

    v_lines = lines[v_idx]
    v_mid_x = 0.5 * (v_lines[:, 0] + v_lines[:, 2])
    left = v_lines[int(np.argmin(v_mid_x))]
    right = v_lines[int(np.argmax(v_mid_x))]
    return far, near, left, right


def compute_court_corners_from_lines(lines10: np.ndarray, strategy: str = "auto"):
    if strategy not in ("auto", "fixed"):
        raise ValueError("strategy must be 'auto' or 'fixed'")

    if strategy == "fixed":
        # Matches the ordering used in estimate_3d_speed_witb.py
        far = lines10[0]
        near = lines10[1]
        left = lines10[3]
        right = lines10[4]
    else:
        # try a couple heuristics; these detectors often output slanted boundaries.
        # If either fails to yield 4 intersections, try the other.
        candidates = [
            pick_court_boundary_auto(lines10),
            pick_court_boundary_by_extremes(lines10),
        ]
        far = near = left = right = None
        for cand in candidates:
            f, n, l, r = cand
            tl = line_intersection(f[:2], f[2:], l[:2], l[2:])
            tr = line_intersection(f[:2], f[2:], r[:2], r[2:])
            bl = line_intersection(n[:2], n[2:], l[:2], l[2:])
            br = line_intersection(n[:2], n[2:], r[:2], r[2:])
            if all(x is not None for x in (tl, tr, br, bl)):
                far, near, left, right = f, n, l, r
                break
        if far is None:
            raise RuntimeError("Failed to pick boundary lines for court corners.")

    far_p1 = far[:2]
    far_p2 = far[2:]
    near_p1 = near[:2]
    near_p2 = near[2:]
    left_p1 = left[:2]
    left_p2 = left[2:]
    right_p1 = right[:2]
    right_p2 = right[2:]

    tl = line_intersection(far_p1, far_p2, left_p1, left_p2)
    tr = line_intersection(far_p1, far_p2, right_p1, right_p2)
    bl = line_intersection(near_p1, near_p2, left_p1, left_p2)
    br = line_intersection(near_p1, near_p2, right_p1, right_p2)

    if any(x is None for x in (tl, tr, br, bl)):
        raise RuntimeError("Failed to compute court corners from detected lines.")

    # order: [tl, tr, br, bl] in image
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def build_initial_K(W: int, H: int, f_scale: float = 1.2):
    """Weak calibration: principal point at center, focal from heuristic."""
    f = float(f_scale) * float(max(W, H))
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def solve_camera_from_court(img_corners: np.ndarray, K: np.ndarray):
    """
    Solve camera extrinsics from 4 court corners on ground plane (Z=0).
    World coords (meters): near baseline y=0, far baseline y=COURT_L.
    """
    world_pts = np.array(
        [
            [0.0, COURT_L, 0.0],
            [COURT_W_DOUBLES, COURT_L, 0.0],
            [COURT_W_DOUBLES, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        world_pts,
        img_corners.astype(np.float64),
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed.")
    return rvec, tvec, dist


def load_ball(ball_path: str):
    b = np.load(ball_path)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"ball npy must be (T,2); got {b.shape}")
    mask = (b[:, 0] != 0) & (b[:, 1] != 0) & np.isfinite(b[:, 0]) & np.isfinite(b[:, 1])
    return b, mask


def remove_spike_outliers(points: np.ndarray, mask: np.ndarray, z: float = 8.0):
    """Mark large inter-frame jumps as missing, then return updated mask."""
    pts = points.copy()
    m = mask.copy()
    idx = np.where(m)[0]
    if idx.size < 10:
        return m

    diffs = np.linalg.norm(np.diff(pts[idx], axis=0), axis=1)
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med)) + 1e-6
    thr = med + z * 1.4826 * mad
    bad = np.where(diffs > thr)[0]
    # mark the later point of the jump as missing
    for b in bad:
        m[idx[b + 1]] = False
    return m


def interpolate_2d(points: np.ndarray, mask: np.ndarray):
    """Linear interpolation over missing frames."""
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
    """Simple median filter smoothing (per-dimension).

    OpenCV medianBlur only supports 8U/16U, so we implement median filtering
    in numpy.
    """
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
    """
    Pick an air-flight segment using acceleration peaks (hits/bounce cause spikes).

    Steps:
    - compute v and a in pixel space
    - find large accel peaks using MAD threshold
    - pick the strongest peak as serve-hit; choose next strong peak after it as end
    """
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
        # fallback: around max speed
        spd = np.linalg.norm(v, axis=1)
        peak = int(np.argmax(spd))
        start = peak
        end = min(T, start + int(round(max_len_s * fps)))
        return start, max(start + 2, end)

    # score peaks by acc magnitude
    peaks = peaks[np.argsort(-acc[peaks])]
    serve_peak = int(peaks[0])
    # a index corresponds to frame i+1 roughly (since a uses v[1:])
    start = max(0, serve_peak - 1)

    min_len = int(round(min_len_s * fps))
    max_len = int(round(max_len_s * fps))
    end_min = min(T, start + max(min_len, 2))
    end_max = min(T, start + max_len)

    # pick next strong peak after start with sufficient separation
    end = None
    for p in peaks[1:]:
        p = int(p)
        # map to frame index similarly
        f = p + 1
        if end_min <= f <= end_max:
            end = f
            break
    if end is None:
        end = end_max
    if end <= start + 1:
        end = min(T, start + 2)
    return start, end


def project_points(cam: Camera, XYZ: np.ndarray):
    img, _ = cv2.projectPoints(XYZ.astype(np.float64), cam.rvec, cam.tvec, cam.K, cam.dist)
    return img.reshape(-1, 2)


def world_from_image_on_ground(cam: Camera, uv: np.ndarray):
    """Back-project image point to ground plane Z=0 via ray-plane intersection."""
    Kinv = np.linalg.inv(cam.K)
    R, _ = cv2.Rodrigues(cam.rvec)
    C = (-R.T @ cam.tvec).reshape(3)  # camera center in world

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float64)], axis=1)
    rays_c = (Kinv @ uv1.T).T
    rays_w = (R.T @ rays_c.T).T
    Cz = C[2]
    rz = rays_w[:, 2]
    s = -Cz / (rz + 1e-12)
    Pw = C[None, :] + s[:, None] * rays_w
    return Pw[:, :2]


def court_world_points_for_lines(court_w: float):
    """
    World points for the 10 important lines in CourtReference.get_important_lines().

    Order of lines is expected to match:
      0 baseline_top, 1 baseline_bottom, 2 net,
      3 left_court_line, 4 right_court_line,
      5 left_inner_line, 6 right_inner_line,
      7 middle_line, 8 top_inner_line, 9 bottom_inner_line

    For each line we return its two endpoints as (2,3). Total (20,3).
    """
    # inner (singles) offsets within doubles
    inner_off = 0.5 * (court_w - COURT_W_SINGLES)
    x_inner_l = inner_off
    x_inner_r = court_w - inner_off

    # key x positions
    x_l = 0.0
    x_r = float(court_w)
    x_m = 0.5 * float(court_w)

    # key y positions
    y_near = 0.0
    y_far = COURT_L
    y_net = NET_Y
    y_s_near = SERVICE_Y_NEAR
    y_s_far = SERVICE_Y_FAR

    pts = []
    # baseline_top (far)
    pts += [[x_l, y_far, 0.0], [x_r, y_far, 0.0]]
    # baseline_bottom (near)
    pts += [[x_l, y_near, 0.0], [x_r, y_near, 0.0]]
    # net
    pts += [[x_l, y_net, 0.0], [x_r, y_net, 0.0]]
    # left/right outer court lines
    pts += [[x_l, y_far, 0.0], [x_l, y_near, 0.0]]
    pts += [[x_r, y_far, 0.0], [x_r, y_near, 0.0]]
    # left/right inner (singles) lines
    pts += [[x_inner_l, y_far, 0.0], [x_inner_l, y_near, 0.0]]
    pts += [[x_inner_r, y_far, 0.0], [x_inner_r, y_near, 0.0]]
    # middle service line (between service lines)
    pts += [[x_m, y_s_far, 0.0], [x_m, y_s_near, 0.0]]
    # top/bottom inner service lines (singles width)
    pts += [[x_inner_l, y_s_far, 0.0], [x_inner_r, y_s_far, 0.0]]
    pts += [[x_inner_l, y_s_near, 0.0], [x_inner_r, y_s_near, 0.0]]
    return np.asarray(pts, dtype=np.float64)


def image_points_from_lines(lines10: np.ndarray):
    """Convert (10,4) lines to (20,2) endpoints in the same order."""
    pts = []
    for l in np.asarray(lines10, dtype=np.float64):
        pts.append([l[0], l[1]])
        pts.append([l[2], l[3]])
    return np.asarray(pts, dtype=np.float64)


def map_detected_lines_to_roles(lines10: np.ndarray, W: int, H: int):
    """
    Map CourtDetector output lines (10,4) to semantic court roles.

    CourtDetector's `find_lines_location()` returns the 10 important lines but the
    *index order is not guaranteed* across forks/versions. For monocular 3D, we
    need a stable correspondence to world court geometry.

    Returns dict with keys:
      baseline_top, baseline_bottom, net,
      left_court_line, right_court_line,
      left_inner_line, right_inner_line,
      middle_line, top_inner_line, bottom_inner_line
    """
    lines = np.asarray(lines10, dtype=np.float32).reshape(-1, 4)
    if lines.shape != (10, 4):
        raise ValueError(f"Expected (10,4), got {lines.shape}")

    ang = np.array([_line_angle_deg(l) for l in lines], dtype=np.float32)
    ln = np.array([_line_len(l) for l in lines], dtype=np.float32)
    mx = 0.5 * (lines[:, 0] + lines[:, 2])
    my = 0.5 * (lines[:, 1] + lines[:, 3])

    horiz_score = np.minimum(np.abs(ang), np.abs(np.abs(ang) - 180.0))
    is_h = horiz_score < 15.0

    cand = np.array([60.0, 90.0, 120.0], dtype=np.float32)[None, :]
    vert_score = np.min(np.abs(np.abs(ang[:, None]) - cand), axis=1)
    is_v = vert_score < 20.0

    # horizontals
    h_idx = np.where(is_h)[0]
    if h_idx.size < 3:
        # fall back to best scored
        h_idx = np.argsort(horiz_score)[:5]
    # far/near baselines are typically among the longest horizontals
    h_long = h_idx[np.argsort(-ln[h_idx])]
    h_take = h_long[: min(4, h_long.size)]
    far_i = int(h_take[np.argmin(my[h_take])])
    near_i = int(h_take[np.argmax(my[h_take])])
    baseline_top = lines[far_i]
    baseline_bottom = lines[near_i]

    remaining_h = [i for i in h_idx.tolist() if i not in (far_i, near_i)]
    if len(remaining_h) == 0:
        raise RuntimeError("Could not identify net/service lines from horizontals")

    mid_y = 0.5 * (float(my[far_i]) + float(my[near_i]))
    net_i = int(min(remaining_h, key=lambda i: abs(float(my[i]) - mid_y)))
    net = lines[net_i]

    remaining_h2 = [i for i in remaining_h if i != net_i]
    # service lines (top_inner_line far side, bottom_inner_line near side)
    if len(remaining_h2) >= 2:
        a, b = remaining_h2[0], remaining_h2[1]
        top_inner_i, bottom_inner_i = (a, b) if my[a] < my[b] else (b, a)
    elif len(remaining_h2) == 1:
        # sometimes only one service line is visible; approximate by position
        only = remaining_h2[0]
        if my[only] < mid_y:
            top_inner_i, bottom_inner_i = only, None
        else:
            top_inner_i, bottom_inner_i = None, only
    else:
        top_inner_i, bottom_inner_i = None, None

    top_inner_line = lines[top_inner_i] if top_inner_i is not None else None
    bottom_inner_line = lines[bottom_inner_i] if bottom_inner_i is not None else None

    # vertical-ish lines
    v_idx = np.where(is_v)[0]
    if v_idx.size < 2:
        raise RuntimeError("Could not identify court boundary lines")

    v_long = v_idx[np.argsort(-ln[v_idx])]
    # take up to 5 vertical-ish candidates
    v_take = v_long[: min(5, v_long.size)]
    left_outer_i = int(v_take[np.argmin(mx[v_take])])
    right_outer_i = int(v_take[np.argmax(mx[v_take])])
    left_court_line = lines[left_outer_i]
    right_court_line = lines[right_outer_i]

    remaining_v = [i for i in v_take.tolist() if i not in (left_outer_i, right_outer_i)]
    cx = W / 2.0
    left_candidates = [i for i in remaining_v if mx[i] < cx]
    right_candidates = [i for i in remaining_v if mx[i] > cx]
    left_inner_i = int(max(left_candidates, key=lambda i: float(mx[i]))) if left_candidates else None
    right_inner_i = int(min(right_candidates, key=lambda i: float(mx[i]))) if right_candidates else None

    used = set([left_outer_i, right_outer_i])
    if left_inner_i is not None:
        used.add(left_inner_i)
    if right_inner_i is not None:
        used.add(right_inner_i)
    middle_candidates = [i for i in v_take.tolist() if i not in used]
    middle_i = int(min(middle_candidates, key=lambda i: abs(float(mx[i]) - cx))) if middle_candidates else None

    left_inner_line = lines[left_inner_i] if left_inner_i is not None else None
    right_inner_line = lines[right_inner_i] if right_inner_i is not None else None
    middle_line = lines[middle_i] if middle_i is not None else None

    roles = {
        "baseline_top": baseline_top,
        "baseline_bottom": baseline_bottom,
        "net": net,
        "left_court_line": left_court_line,
        "right_court_line": right_court_line,
        "left_inner_line": left_inner_line,
        "right_inner_line": right_inner_line,
        "middle_line": middle_line,
        "top_inner_line": top_inner_line,
        "bottom_inner_line": bottom_inner_line,
    }

    missing = [k for k, v in roles.items() if v is None]
    if missing:
        # don't hard-fail: camera refinement can run with a subset if needed
        pass
    return roles


def image_points_from_roles(roles: dict) -> np.ndarray:
    """Build (N,2) image points from role-ordered lines, skipping missing roles."""
    order = [
        "baseline_top",
        "baseline_bottom",
        "net",
        "left_court_line",
        "right_court_line",
        "left_inner_line",
        "right_inner_line",
        "middle_line",
        "top_inner_line",
        "bottom_inner_line",
    ]
    pts = []
    kept = []
    for k in order:
        l = roles.get(k)
        if l is None:
            continue
        pts.append([float(l[0]), float(l[1])])
        pts.append([float(l[2]), float(l[3])])
        kept.append(k)
    return np.asarray(pts, dtype=np.float64), kept


def refine_camera_from_court_points(
    img_pts: np.ndarray,
    world_pts: np.ndarray,
    W: int,
    H: int,
    f_scale: float = 1.2,
):
    """Refine focal length + extrinsics by minimizing court reprojection error."""
    if img_pts.ndim != 2 or img_pts.shape[1] != 2:
        raise ValueError(f"Expected img_pts (N,2), got {img_pts.shape}")
    if world_pts.ndim != 2 or world_pts.shape[1] != 3:
        raise ValueError(f"Expected world_pts (N,3), got {world_pts.shape}")
    if img_pts.shape[0] != world_pts.shape[0]:
        raise ValueError("img_pts and world_pts must have same N")
    if img_pts.shape[0] < 8:
        raise ValueError("Need at least 8 point correspondences for stable refinement")

    cx = W / 2.0
    cy = H / 2.0
    f0 = float(f_scale) * float(max(W, H))
    K0 = np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec0, tvec0 = cv2.solvePnP(
        world_pts,
        img_pts,
        K0,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Initial solvePnP failed for camera refinement")

    # Optimize log_f (keeps f>0), rvec, tvec
    p0 = np.concatenate([
        np.array([math.log(max(f0, 1.0))], dtype=np.float64),
        rvec0.reshape(3),
        tvec0.reshape(3),
    ])

    f_lo = 0.4 * float(max(W, H))
    f_hi = 6.0 * float(max(W, H))
    lb = np.concatenate([
        np.array([math.log(f_lo)], dtype=np.float64),
        np.full(3, -np.inf, dtype=np.float64),
        np.full(3, -np.inf, dtype=np.float64),
    ])
    ub = np.concatenate([
        np.array([math.log(f_hi)], dtype=np.float64),
        np.full(3, np.inf, dtype=np.float64),
        np.full(3, np.inf, dtype=np.float64),
    ])

    def residual(p: np.ndarray):
        log_f = float(p[0])
        f = float(math.exp(log_f))
        rvec = p[1:4].reshape(3, 1)
        tvec = p[4:7].reshape(3, 1)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
        uv = proj.reshape(-1, 2)
        return (uv - img_pts).reshape(-1)

    res = least_squares(
        residual,
        p0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=2.0,
        max_nfev=200,
    )
    log_f = float(res.x[0])
    f = float(math.exp(log_f))
    rvec = res.x[1:4].reshape(3, 1)
    tvec = res.x[4:7].reshape(3, 1)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Camera(K=K, dist=dist, rvec=rvec, tvec=tvec, W=W, H=H), res


def pick_fit_segment_from_speed(XY: np.ndarray, fps: float, window_s: float = 0.35):
    """Pick a fitting window right after the fastest motion peak."""
    dt = 1.0 / fps
    d = XY[1:] - XY[:-1]
    speed = np.linalg.norm(d, axis=1) / dt

    # avoid choosing edge peaks (often due to bad interpolation at ends)
    if speed.shape[0] >= 10:
        lo = int(0.05 * speed.shape[0])
        hi = int(0.95 * speed.shape[0])
        peak = int(lo + np.argmax(speed[lo:hi]))
    else:
        peak = int(np.argmax(speed))

    win = int(round(window_s * fps))
    win = max(win, int(0.20 * fps))
    start = max(0, peak)
    end = min(len(XY), start + win)
    if end - start < 2:
        end = min(len(XY), start + 2)
    return start, end, speed


def fit_3d_projectile(cam: Camera, uv_obs: np.ndarray, t: np.ndarray, XY_init: np.ndarray):
    """
    Fit ballistic parameters by minimizing reprojection error.
    Params: x0,y0,z0,vx,vy,vz in world meters, m/s.
    """
    if uv_obs.shape[0] != t.shape[0] or uv_obs.shape[0] != XY_init.shape[0]:
        raise ValueError("uv_obs, t, XY_init must share the same length")

    x0, y0 = map(float, XY_init[0])
    if len(t) > 1:
        dt = float(t[1] - t[0])
        vxy = (XY_init[1] - XY_init[0]) / max(dt, 1e-6)
        vx0, vy0 = float(vxy[0]), float(vxy[1])
    else:
        vx0, vy0 = 0.0, 0.0

    z0 = 1.0
    vz0 = 0.0
    p0 = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float64)

    xy_min = XY_init.min(axis=0)
    xy_max = XY_init.max(axis=0)

    # prefer solutions near the court; allow some margin behind baselines.
    pad = 8.0
    x_min_phys = -2.0
    x_max_phys = COURT_W_DOUBLES + 2.0
    y_min_phys = -6.0
    y_max_phys = COURT_L + 6.0
    # keep speeds in a physically plausible range; serves rarely exceed ~75 m/s
    vmax = 80.0
    lb_xy = np.maximum([xy_min[0] - pad, xy_min[1] - pad], [x_min_phys, y_min_phys])
    ub_xy = np.minimum([xy_max[0] + pad, xy_max[1] + pad], [x_max_phys, y_max_phys])
    lb = np.array([lb_xy[0], lb_xy[1], 0.0, -vmax, -vmax, -vmax], dtype=np.float64)
    ub = np.array([ub_xy[0], ub_xy[1], 20.0, vmax, vmax, vmax], dtype=np.float64)
    p0 = np.minimum(np.maximum(p0, lb + 1e-6), ub - 1e-6)

    def residual(p: np.ndarray):
        x0, y0, z0, vx, vy, vz = p
        X = x0 + vx * t
        Y = y0 + vy * t
        Z = z0 + vz * t - 0.5 * G * t * t
        XYZ = np.stack([X, Y, Z], axis=1)
        uv_pred = project_points(cam, XYZ)

        # soft constraint: keep Z above ground in fitted window
        # Penalize ground penetration. Hinge is fine, but penalize more strongly.
        z_pen = np.minimum(Z, 0.0)  # negative values only
        lam = 400.0

        # soft constraint: keep (X,Y) near the court area (avoid wild extrinsics solutions)
        x_pen = np.maximum(x_min_phys - X, 0.0) + np.maximum(X - x_max_phys, 0.0)
        y_pen = np.maximum(y_min_phys - Y, 0.0) + np.maximum(Y - y_max_phys, 0.0)
        lam_xy = 0.05
        return np.concatenate([
            (uv_pred - uv_obs).reshape(-1),
            math.sqrt(lam) * z_pen,
            math.sqrt(lam_xy) * x_pen,
            math.sqrt(lam_xy) * y_pen,
        ])

    res = least_squares(
        residual,
        p0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=3.0,
        max_nfev=300,
    )
    return res.x, res


def reconstruct(
    video_path: str,
    ball_npy: str,
    line_npy: str,
    out_npz: str,
    window_s: float = 0.35,
    corners_strategy: str = "auto",
    f_scale: float = 1.2,
    start: Optional[int] = None,
    end: Optional[int] = None,
):
    fps, W, H = read_video_meta(video_path)

    ball_raw, mask = load_ball(ball_npy)
    mask = remove_spike_outliers(ball_raw, mask)
    ball_uv = interpolate_2d(ball_raw, mask)
    ball_uv = smooth_2d(ball_uv, ksize=7)

    lines = np.load(line_npy, allow_pickle=True)
    if len(lines) < 1:
        raise RuntimeError("line npy is empty")
    lines0 = reshape_lines(lines[0])
    img_corners = compute_court_corners_from_lines(lines0, strategy=corners_strategy)

    # Camera estimation: build stable correspondences by mapping detected lines to roles.
    # If refinement fails, fall back to corners-only PnP.
    roles = map_detected_lines_to_roles(lines0, W=W, H=H)
    img_pts, kept_roles = image_points_from_roles(roles)
    all_world_pts = court_world_points_for_lines(COURT_W_DOUBLES)

    # world points must match the kept roles order
    role_to_pair_idx = {
        "baseline_top": 0,
        "baseline_bottom": 1,
        "net": 2,
        "left_court_line": 3,
        "right_court_line": 4,
        "left_inner_line": 5,
        "right_inner_line": 6,
        "middle_line": 7,
        "top_inner_line": 8,
        "bottom_inner_line": 9,
    }
    wp = []
    for r in kept_roles:
        k = role_to_pair_idx[r]
        wp.append(all_world_pts[2 * k])
        wp.append(all_world_pts[2 * k + 1])
    lines_world_pts = np.asarray(wp, dtype=np.float64)
    try:
        cam, cam_opt = refine_camera_from_court_points(
            img_pts=img_pts,
            world_pts=lines_world_pts,
            W=W,
            H=H,
            f_scale=f_scale,
        )
    except Exception:
        K = build_initial_K(W, H, f_scale=f_scale)
        rvec, tvec, dist = solve_camera_from_court(img_corners, K)
        cam = Camera(K=K, dist=dist, rvec=rvec, tvec=tvec, W=W, H=H)
        cam_opt = None

    # Choose the fitting segment.
    # Prefer accel-peak based flight segmentation (hit -> bounce) in pixel space.
    seg_start, seg_end = pick_flight_segment_from_accel(
        ball_uv,
        fps,
        min_len_s=max(0.20, 0.5 * window_s),
        max_len_s=max(0.60, 1.6 * window_s),
    )

    XY_ground = world_from_image_on_ground(cam, ball_uv)
    # for reporting only
    _, _, speed_ground = pick_fit_segment_from_speed(XY_ground, fps, window_s=window_s)
    if start is not None:
        seg_start = int(start)
    if end is not None:
        seg_end = int(end)
    seg_start = max(0, min(seg_start, len(ball_uv) - 1))
    seg_end = max(seg_start + 1, min(seg_end, len(ball_uv)))

    uv_seg = ball_uv[seg_start:seg_end]
    XY_seg = XY_ground[seg_start:seg_end]
    t = (np.arange(seg_start, seg_end) - seg_start) / fps

    params, opt = fit_3d_projectile(cam, uv_seg, t, XY_seg)
    x0, y0, z0, vx, vy, vz = map(float, params)

    v0 = math.sqrt(vx * vx + vy * vy + vz * vz)
    vxy0 = math.sqrt(vx * vx + vy * vy)

    X = x0 + vx * t
    Y = y0 + vy * t
    Z = z0 + vz * t - 0.5 * G * t * t
    XYZ = np.stack([X, Y, Z], axis=1)
    uv_pred = project_points(cam, XYZ)

    np.savez(
        out_npz,
        fps=fps,
        W=W,
        H=H,
        segment_start=seg_start,
        segment_end=seg_end,
        params=params,
        speed_3d_mps=v0,
        speed_3d_kmh=v0 * 3.6,
        speed_ground_mps=vxy0,
        speed_ground_kmh=vxy0 * 3.6,
        uv_obs=uv_seg,
        uv_pred=uv_pred,
        XY_ground=XY_seg,
        XYZ=XYZ,
        img_corners=img_corners,
        K=cam.K,
        rvec=cam.rvec,
        tvec=cam.tvec,
        corners_strategy=corners_strategy,
        f_scale=f_scale,
        opt_success=bool(opt.success),
        opt_cost=float(opt.cost),
        opt_nfev=int(opt.nfev),
        speed_ground_series=speed_ground,
        cam_refine_success=(cam_opt.success if cam_opt is not None else False),
        cam_refine_cost=(float(cam_opt.cost) if cam_opt is not None else -1.0),
    )

    return {
        "fps": fps,
        "W": W,
        "H": H,
        "segment": (seg_start, seg_end),
        "params": params,
        "speed_3d_mps": v0,
        "speed_ground_mps": vxy0,
        "opt_success": bool(opt.success),
        "opt_cost": float(opt.cost),
    }


def main():
    ap = argparse.ArgumentParser(description="Monocular 2D->3D tennis ball trajectory reconstruction")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--ball", required=True, help="Ball detections .npy, shape (T,2) pixels")
    ap.add_argument("--line", required=True, help="Court lines .npy from CourtDetector")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--window_s", type=float, default=0.35, help="Seconds to fit after peak speed")
    ap.add_argument(
        "--corners_strategy",
        choices=["auto", "fixed"],
        default="auto",
        help="How to choose boundary lines for corners",
    )
    ap.add_argument("--f_scale", type=float, default=1.2, help="Focal length scale vs max(W,H)")
    ap.add_argument("--start", type=int, default=None, help="Override segment start frame (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Override segment end frame (exclusive)")
    args = ap.parse_args()

    info = reconstruct(
        video_path=args.video,
        ball_npy=args.ball,
        line_npy=args.line,
        out_npz=args.out,
        window_s=args.window_s,
        corners_strategy=args.corners_strategy,
        f_scale=args.f_scale,
        start=args.start,
        end=args.end,
    )
    print("=== 2D->3D trajectory reconstruction ===")
    print(f"fps={info['fps']:.3f}, video(W,H)=({info['W']},{info['H']})")
    print(f"segment frames: [{info['segment'][0]}, {info['segment'][1]})")
    p = info["params"]
    print(
        "params [x0 y0 z0 vx vy vz] = "
        f"[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f} {p[4]:.3f} {p[5]:.3f}]"
    )
    print(f"speed_3d = {info['speed_3d_mps']:.3f} m/s  ({info['speed_3d_mps']*3.6:.1f} km/h)")
    print(f"speed_ground = {info['speed_ground_mps']:.3f} m/s  ({info['speed_ground_mps']*3.6:.1f} km/h)")
    print(f"opt success={info['opt_success']}, cost={info['opt_cost']:.3f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
