import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


# ---------------------------
# Court geometry (singles)
# ---------------------------
COURT_W = 8.23   # meters (singles width)
COURT_L = 23.77  # meters (court length)
G = 9.81


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    W: int
    H: int


def reshape_lines(lines_1d: np.ndarray) -> np.ndarray:
    """(40,) -> (10,4) float32"""
    arr = np.asarray(lines_1d, dtype=np.float32).reshape(-1, 4)
    return arr


def line_intersection(p1, p2, p3, p4, eps=1e-6):
    """Intersect infinite lines p1-p2 and p3-p4. Return (x,y) or None if parallel."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < eps:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return np.array([px, py], dtype=np.float32)


def pick_court_boundary(lines10: np.ndarray):
    """
    Your CourtDetector output seems to include:
      - far baseline (top)  : (x,y)-(x,y)
      - near baseline (bottom)
      - left boundary line
      - right boundary line
    From your sample, indices likely:
      0: far baseline
      1: near baseline
      3: left boundary
      4: right boundary
    We'll use these by default.
    """
    far = lines10[0]
    near = lines10[1]
    left = lines10[3]
    right = lines10[4]
    return far, near, left, right


def compute_court_corners_from_lines(lines10: np.ndarray):
    far, near, left, right = pick_court_boundary(lines10)

    far_p1 = far[:2]; far_p2 = far[2:]
    near_p1 = near[:2]; near_p2 = near[2:]
    left_p1 = left[:2]; left_p2 = left[2:]
    right_p1 = right[:2]; right_p2 = right[2:]

    tl = line_intersection(far_p1, far_p2, left_p1, left_p2)
    tr = line_intersection(far_p1, far_p2, right_p1, right_p2)
    bl = line_intersection(near_p1, near_p2, left_p1, left_p2)
    br = line_intersection(near_p1, near_p2, right_p1, right_p2)

    if any(x is None for x in [tl, tr, bl, br]):
        raise RuntimeError("Failed to compute court corners: some boundary lines are parallel.")

    # order: [tl, tr, br, bl] in image
    img_pts = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
    return img_pts


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
    return fps, W, H


def build_initial_K(W: int, H: int):
    """
    Weak calibration: assume principal point at image center.
    f init: 1.2 * max(W,H) is a common heuristic for broadcast-like FOV.
    """
    f = 1.2 * float(max(W, H))
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def solve_camera_from_court(img_corners: np.ndarray, K: np.ndarray):
    """
    Use solvePnP with court corners on Z=0 plane.
    World coords (meters): define near baseline y=0, far baseline y=COURT_L.
    Order must match img_pts: tl,tr,br,bl corresponds to far-left, far-right, near-right, near-left.
    """
    world_pts = np.array([
        [0.0, COURT_L, 0.0],        # far-left  (tl)
        [COURT_W, COURT_L, 0.0],    # far-right (tr)
        [COURT_W, 0.0, 0.0],        # near-right (br)
        [0.0, 0.0, 0.0],            # near-left (bl)
    ], dtype=np.float64)

    dist = np.zeros((4, 1), dtype=np.float64)

    # use iterative PnP
    ok, rvec, tvec = cv2.solvePnP(world_pts, img_corners.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed.")
    return rvec, tvec, dist


def load_ball(ball_path: str):
    b = np.load(ball_path)
    b = b.astype(np.float64)
    # mark missing
    mask = (b[:, 0] != 0) & (b[:, 1] != 0)
    return b, mask


def interpolate_2d(points: np.ndarray, mask: np.ndarray):
    """
    Linear interpolation over missing frames.
    points: (T,2)
    mask: (T,)
    """
    T = points.shape[0]
    out = points.copy()
    idx = np.arange(T)
    for d in range(2):
        valid = mask & np.isfinite(points[:, d])
        if valid.sum() < 5:
            raise RuntimeError("Too few valid ball detections to interpolate.")
        out[:, d] = np.interp(idx, idx[valid], points[valid, d])
    return out


def world_from_image_on_ground(cam: Camera, uv: np.ndarray):
    """
    Back-project an image point to the ground plane Z=0 using ray-plane intersection.
    Uses camera extrinsics and intrinsics.
    """
    Kinv = np.linalg.inv(cam.K)
    R, _ = cv2.Rodrigues(cam.rvec)
    C = (-R.T @ cam.tvec).reshape(3)  # camera center in world

    # uv to normalized ray in camera frame
    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float64)], axis=1)  # (N,3)
    rays_c = (Kinv @ uv1.T).T  # (N,3)
    # to world frame
    rays_w = (R.T @ rays_c.T).T  # (N,3)

    # intersect with plane Z=0 => C.z + s*r.z = 0 => s = -Cz/rz
    Cz = C[2]
    rz = rays_w[:, 2]
    s = -Cz / (rz + 1e-12)
    Pw = C[None, :] + s[:, None] * rays_w
    return Pw[:, :2]  # (N,2) X,Y


def pick_fast_segment(XY: np.ndarray, fps: float, window_s=0.35):
    """
    Find a segment right after the fastest motion on ground plane.
    Returns (start_idx, end_idx) inclusive of start, exclusive of end.
    """
    dt = 1.0 / fps
    d = XY[1:] - XY[:-1]
    speed = np.linalg.norm(d, axis=1) / dt  # m/s (ground-plane)
    # robust: pick top peak index
    peak = int(np.argmax(speed))
    win = int(round(window_s * fps))
    start = max(0, peak)
    end = min(len(XY), start + win)
    if end - start < int(0.15 * fps):
        end = min(len(XY), start + int(0.25 * fps))
    return start, end, speed


def project_points(cam: Camera, XYZ: np.ndarray):
    """
    Project world points XYZ (N,3) to image uv (N,2).
    """
    img, _ = cv2.projectPoints(XYZ.astype(np.float64), cam.rvec, cam.tvec, cam.K, cam.dist)
    return img.reshape(-1, 2)


def fit_3d_projectile(cam: Camera, uv_obs: np.ndarray, t: np.ndarray, XY_init: np.ndarray):
    """
    WITB-style: fit projectile parameters by minimizing reprojection error.

    Params:
      x0,y0,z0,vx,vy,vz  (world coords, meters, m/s)
    We initialize x0,y0 from ground-plane backprojection; z0 with a small positive value.
    """
    assert uv_obs.shape[0] == t.shape[0] == XY_init.shape[0]

    # Initial guess from ground-plane XY and finite differences
    x0, y0 = XY_init[0]
    dt = t[1] - t[0] if len(t) > 1 else 1.0 / 30.0
    vxy = (XY_init[1] - XY_init[0]) / dt if len(t) > 1 else np.array([0.0, 0.0])
    vx0, vy0 = float(vxy[0]), float(vxy[1])

    # z init (meters): start around 1m; vz init 0
    z0 = 1.0
    vz0 = 0.0

    p0 = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float64)

    # bounds (keep reasonable)
    # x,y within court-ish; z >= 0; speeds within plausible range (0..80 m/s ~ 288 km/h)

    # bounds: adapt to current XY to avoid infeasible init due to weak calibration
    xy_min = XY_init.min(axis=0)
    xy_max = XY_init.max(axis=0)
    pad = 30.0  # meters, generous
    lb = np.array([xy_min[0]-pad, xy_min[1]-pad, 0.0, -120, -120, -120], dtype=np.float64)
    ub = np.array([xy_max[0]+pad, xy_max[1]+pad, 20.0,  120,  120,  120], dtype=np.float64)

# also make sure p0 is inside bounds (clip)
p0 = np.minimum(np.maximum(p0, lb + 1e-6), ub - 1e-6)
    def residual(p):
        x0, y0, z0, vx, vy, vz = p
        X = x0 + vx * t
        Y = y0 + vy * t
        Z = z0 + vz * t - 0.5 * G * t * t
        XYZ = np.stack([X, Y, Z], axis=1)
        uv_pred = project_points(cam, XYZ)
        r = (uv_pred - uv_obs).reshape(-1)
        return r

    res = least_squares(residual, p0, bounds=(lb, ub), loss="huber", f_scale=3.0, max_nfev=200)
    p = res.x
    return p, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path (same used to produce npy)")
    ap.add_argument("--ball", default="./output/ball/001.npy")
    ap.add_argument("--line", default="./output/line/001.npy")
    ap.add_argument("--out", default="./output/speed_3d_001.npz")
    ap.add_argument("--window_s", type=float, default=0.35, help="seconds to fit after peak speed")
    args = ap.parse_args()

    fps, W, H = read_video_meta(args.video)

    # Load data
    ball_raw, mask = load_ball(args.ball)
    ball_uv = interpolate_2d(ball_raw, mask)  # (T,2)

    L = np.load(args.line, allow_pickle=True)
    lines0 = reshape_lines(L[0])  # use first frame for stable court
    img_corners = compute_court_corners_from_lines(lines0)  # (4,2) tl,tr,br,bl

    # Camera init
    K = build_initial_K(W, H)
    rvec, tvec, dist = solve_camera_from_court(img_corners, K)
    cam = Camera(K=K, dist=dist, rvec=rvec, tvec=tvec, W=W, H=H)

    # Backproject to ground plane for XY init + segment selection
    XY = world_from_image_on_ground(cam, ball_uv)  # (T,2) meters
    start, end, speed2d = pick_fast_segment(XY, fps, window_s=args.window_s)

    uv_seg = ball_uv[start:end]
    XY_seg = XY[start:end]
    t = (np.arange(start, end) - start) / fps

    # Fit 3D projectile parameters
    p, opt = fit_3d_projectile(cam, uv_seg, t, XY_seg)
    x0, y0, z0, vx, vy, vz = p

    v0 = math.sqrt(vx*vx + vy*vy + vz*vz)
    v0_kmh = v0 * 3.6
    vxy0 = math.sqrt(vx*vx + vy*vy)
    vxy0_kmh = vxy0 * 3.6

    print("=== 3D speed estimation (WITB-style optimization) ===")
    print(f"fps={fps:.3f}, video(W,H)=({W},{H})")
    print(f"segment frames: [{start}, {end})  length={end-start}")
    print(f"init pos (m): x0={x0:.3f} y0={y0:.3f} z0={z0:.3f}")
    print(f"init vel (m/s): vx={vx:.3f} vy={vy:.3f} vz={vz:.3f}")
    print(f"speed_3d = {v0:.3f} m/s  ({v0_kmh:.1f} km/h)")
    print(f"speed_ground = {vxy0:.3f} m/s  ({vxy0_kmh:.1f} km/h)")
    print(f"opt success={opt.success}, cost={opt.cost:.3f}, nfev={opt.nfev}")

    # Reconstruct fitted 3D trajectory for the fitted window
    X = x0 + vx * t
    Y = y0 + vy * t
    Z = z0 + vz * t - 0.5 * G * t * t
    XYZ = np.stack([X, Y, Z], axis=1)
    uv_pred = project_points(cam, XYZ)

    np.savez(
        args.out,
        fps=fps, W=W, H=H,
        segment_start=start, segment_end=end,
        params=p,
        speed_3d_mps=v0, speed_3d_kmh=v0_kmh,
        speed_ground_mps=vxy0, speed_ground_kmh=vxy0_kmh,
        uv_obs=uv_seg, uv_pred=uv_pred,
        XY_ground=XY_seg, XYZ=XYZ,
        img_corners=img_corners,
        K=cam.K, rvec=cam.rvec, tvec=cam.tvec
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()