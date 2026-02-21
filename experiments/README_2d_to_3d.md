# 2D -> 3D Trajectory Reconstruction (Experiments)

This folder contains experimental scripts for building a 3D ball trajectory from 2D detections.

## What This Does

Given:

- `ball.npy`: per-frame 2D ball center in pixels, shape `(T, 2)`.
- `line.npy`: per-frame court lines from the repo's `CourtDetector`.
- `video.mp4`: original video (used for FPS and image size).

We:

- estimate a camera pose from the court corners (weak intrinsics + PnP)
- fit a ballistic (projectile) 3D trajectory by minimizing reprojection error
- save reconstructed `XYZ` plus camera params to an `.npz`

This is a **monocular** reconstruction. Absolute depth scale comes from court dimensions, but accuracy depends on court line detection quality and the weak camera calibration assumption.

## Environment

The script requires (at least):

- `numpy`
- `opencv-python` (provides `cv2`)
- `scipy`

If you use conda:

```bash
conda create -n tennis3d python=3.10 -y
conda activate tennis3d
pip install numpy opencv-python scipy
```

## Run

```bash
python experiments/reconstruct_2d_to_3d.py \
  --video videos/001.mp4 \
  --ball output/ball/001.npy \
  --line output/line/001.npy \
  --out output/traj3d_001.npz
```

## Tips

- If the court line order is stable in your data, try `--corners_strategy fixed`.
- If results look unstable, adjust `--f_scale` (focal length heuristic).
- For debugging, force a segment with `--start/--end`.
