import argparse
import csv
import importlib
import json
import os
import site
import subprocess
import sys
from pathlib import Path

from estimate_pose import dump_pose_from_video


def stage_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_paths(label: str, paths: list[Path]) -> None:
    print(label)
    for p in paths:
        print(f"  - {p}")


def build_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(env.get("CONDA_PREFIX", sys.prefix))

    # Discover nvidia runtime libs from current environment dynamically.
    nvidia_lib_dirs: list[Path] = []
    site_dirs = [Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(Path(user_site))

    seen = set()
    for sp in site_dirs:
        nvidia_root = sp / "nvidia"
        if not nvidia_root.exists() or not nvidia_root.is_dir():
            continue
        for pkg_dir in nvidia_root.iterdir():
            lib_dir = pkg_dir / "lib"
            if lib_dir.exists() and lib_dir.is_dir():
                key = str(lib_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    nvidia_lib_dirs.append(lib_dir)

    # Keep env-level lib as fallback for conda-provided shared libs.
    nvidia_lib_dirs.append(prefix / "lib")

    existing = [str(p) for p in nvidia_lib_dirs if p.exists()]
    if existing:
        env["LD_LIBRARY_PATH"] = ":".join(existing + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def run_cmd(
    stage_name: str,
    cmd: list[str],
    cwd: Path,
    expected_outputs: list[Path] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    stage_header(f"[STAGE] {stage_name}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
    if expected_outputs:
        print("[CHECK] Outputs:")
        for p in expected_outputs:
            print(f"  - {'OK' if p.exists() else 'MISSING'}: {p}")


def read_hit_bounce(csv_path: Path) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"Empty hit/bounce csv: {csv_path}")
    return int(row["hit"]), int(row["bounce"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full serve-speed pipeline for one video id")
    parser.add_argument("--video_id", type=str, default="001")
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="checkpoints/tracknet-v4_best-model.pth")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--server_id", type=int, default=1)
    parser.add_argument("--z_hit", type=float, default=2.775)
    parser.add_argument("--z_bounce", type=float, default=0.0)
    parser.add_argument("--skip_predict", action="store_true", help="Reuse existing ball csv")
    parser.add_argument("--skip_pose", action="store_true", help="Reuse existing pose dump")
    parser.add_argument("--gpu_only", action="store_true", help="Fail immediately if CUDA is unavailable")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    video_path = Path(args.video_path) if args.video_path else (root / "videos" / f"{args.video_id}.mp4")
    if not video_path.is_absolute():
        video_path = root / video_path

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path

    ball_csv = root / "output" / "ball" / f"{args.video_id}_predict_ball.csv"
    line_npy = root / "output" / "line" / f"{args.video_id}.npy"
    hit_bounce_csv = root / "output" / "hit_bounce" / f"{args.video_id}.csv"
    pose_dir = root / "output" / "pose_keypoints" / args.video_id
    pose_npz = pose_dir / "dump.npz"
    players_npy = root / "output" / "pose_keypoints" / args.video_id / "2_keypoints.npy"
    step1_json = root / "output" / "step1_2d" / args.video_id / "step1_2d.json"
    step2_json = root / "output" / "step2_velocity" / args.video_id / "step2_velocity.json"

    step2_traj_csv = root / "output" / "step2_trajectory" / args.video_id / "step2_trajectory.csv"
    step2_traj_png = root / "output" / "step2_trajectory" / args.video_id / "step2_trajectory.png"

    stage_header("[PIPELINE START]")
    print(f"video_id   : {args.video_id}")
    print(f"video_path : {video_path}")
    print(f"device     : {args.device}")
    print(f"server_id  : {args.server_id}")
    print(f"z_hit      : {args.z_hit}")
    print(f"z_bounce   : {args.z_bounce}")
    print(f"gpu_only   : {args.gpu_only}")

    run_env = os.environ.copy()
    if args.device == "cuda" or args.gpu_only:
        run_env = build_gpu_env()
        os.environ.update(run_env)
        try:
            ort = importlib.import_module("onnxruntime")

            providers = ort.get_available_providers()
            print(f"onnxruntime providers: {providers}")
            if args.gpu_only and "CUDAExecutionProvider" not in providers:
                raise RuntimeError("CUDAExecutionProvider is not available in onnxruntime")
        except Exception as e:
            if args.gpu_only:
                raise RuntimeError(f"GPU-only mode failed: {e}") from e
            print(f"[WARN] CUDA precheck warning: {e}")

    print_paths("[EXPECTED INPUTS]", [video_path, model_path])
    print_paths(
        "[EXPECTED OUTPUTS]",
        [
            ball_csv,
            line_npy,
            players_npy,
            hit_bounce_csv,
            step1_json,
            step2_json,
            step2_traj_csv,
            step2_traj_png,
        ],
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not args.skip_predict:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        run_cmd(
            "Ball detection (predict.py)",
            [
                sys.executable,
                "predict.py",
                "--video_path",
                str(video_path),
                "--model_path",
                str(model_path),
                "--only_csv",
                "--device",
                args.device,
            ],
            root,
            expected_outputs=[ball_csv],
            env=run_env,
        )

    if not ball_csv.exists():
        raise FileNotFoundError(f"Ball csv not found: {ball_csv}")

    run_cmd(
        "Court line detection (detect_line_for_video.py)",
        [
            sys.executable,
            "detect_line_for_video.py",
            "--video_path",
            str(video_path),
            "--out_npy",
            str(line_npy),
        ],
        root,
        expected_outputs=[line_npy],
        env=run_env,
    )

    if not args.skip_pose:
        pose_device = "cuda" if (args.device == "cuda" or args.gpu_only) else "cpu"
        pose_code = (
            "from estimate_pose import dump_pose_from_video; "
            f"dump_pose_from_video(video_path={str(video_path)!r}, "
            f"out_dir={str(pose_dir)!r}, "
            f"device={pose_device!r}, "
            "backend='onnxruntime', mode='performance', to_openpose=False, max_frames=-1)"
        )
        run_cmd(
            f"Pose estimation (RTMLib, device={pose_device})",
            [
                sys.executable,
                "-c",
                pose_code,
            ],
            root,
            expected_outputs=[pose_npz],
            env=run_env,
        )

    if not pose_npz.exists():
        raise FileNotFoundError(f"Pose npz not found: {pose_npz}")

    run_cmd(
        "Two-player filter (pose_filter.py)",
        [
            sys.executable,
            "pose_filter.py",
            "--video_id",
            args.video_id,
            "--pose_npz",
            str(pose_npz),
            "--line_npy",
            str(line_npy),
            "--video_path",
            str(video_path),
        ],
        root,
        expected_outputs=[players_npy],
        env=run_env,
    )

    run_cmd(
        "Hit/Bounce detection (hit_bounce.py)",
        [
            sys.executable,
            "hit_bounce.py",
            "--video",
            str(video_path),
            "--ball",
            str(ball_csv),
            "--players",
            str(players_npy),
            "--out",
            str(hit_bounce_csv),
        ],
        root,
        expected_outputs=[hit_bounce_csv],
        env=run_env,
    )

    hit_frame, bounce_frame = read_hit_bounce(hit_bounce_csv)
    print(f"\n[INFO] hit_frame={hit_frame}, bounce_frame={bounce_frame}")

    run_cmd(
        "Step1 coordinate mapping (step1_standard_2d.py)",
        [
            sys.executable,
            "step1_standard_2d.py",
            "--video_id",
            args.video_id,
            "--hit_frame",
            str(hit_frame),
            "--bounce_frame",
            str(bounce_frame),
            "--server_id",
            str(args.server_id),
            "--video_path",
            str(video_path),
            "--line_npy",
            str(line_npy),
            "--players_npy",
            str(players_npy),
            "--ball_csv",
            str(ball_csv),
        ],
        root,
        expected_outputs=[step1_json],
        env=run_env,
    )

    run_cmd(
        "Step2 velocity solve (step2_initial_velocity.py)",
        [
            sys.executable,
            "step2_initial_velocity.py",
            "--video_id",
            args.video_id,
            "--step1_json",
            str(step1_json),
            "--z_hit",
            str(args.z_hit),
            "--z_bounce",
            str(args.z_bounce),
        ],
        root,
        expected_outputs=[step2_json, step2_traj_csv, step2_traj_png],
        env=run_env,
    )

    if not step2_json.exists():
        raise FileNotFoundError(f"Step2 output not found: {step2_json}")

    with step2_json.open("r", encoding="utf-8") as f:
        result = json.load(f)

    v0 = result["v0_m_s"]
    print("\n[DONE] Serve speed computed")
    print(f"video_id          : {args.video_id}")
    print(f"speed (m/s)       : {v0['speed']:.4f}")
    print(f"speed (km/h)      : {v0['speed_kmh']:.2f}")
    print(f"v0 (vx,vy,vz)     : ({v0['vx']:.4f}, {v0['vy']:.4f}, {v0['vz']:.4f})")
    print(f"hit/bounce csv    : {hit_bounce_csv}")
    print(f"step1 json        : {step1_json}")
    print(f"step2 json        : {step2_json}")
    print(f"step2 traj csv    : {step2_traj_csv}")
    print(f"step2 traj png    : {step2_traj_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
