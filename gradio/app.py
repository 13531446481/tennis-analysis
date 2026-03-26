import csv
import json
import os
import site
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
STAGE_KEYS = ["prepare", "ball", "line", "pose", "filter", "hit_bounce", "step1", "step2"]
STAGE_TITLES = [
    "准备会话",
    "球检测",
    "场地线检测",
    "姿态估计",
    "双人筛选",
    "击球/落地",
    "二维映射",
    "速度求解",
]
DEFAULT_MODEL_PATH = "checkpoints/tracknet-v4_best-model.pth"

THEME = gr.themes.Origin(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    text_size="sm",
)

CSS = (ROOT / "gradio" / "styles.css").read_text(encoding="utf-8")



@dataclass
class SessionState:
    video_id: str = ""
    video_path: str = ""
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "cuda"
    server_id: int = 1
    z_hit: float = 2.775
    z_bounce: float = 0.0
    ball_csv: str = ""
    line_npy: str = ""
    pose_dir: str = ""
    pose_npz: str = ""
    players_npy: str = ""
    hit_bounce_csv: str = ""
    step1_json: str = ""
    step2_json: str = ""
    step2_traj_csv: str = ""
    step2_traj_png: str = ""
    last_hit: str = ""
    last_bounce: str = ""
    last_speed_kmh: str = ""
    current_stage: int = 0
    completed: dict[str, bool] = field(default_factory=dict)
    log: str = ""


def build_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(env.get("CONDA_PREFIX", sys.prefix))
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
    nvidia_lib_dirs.append(prefix / "lib")
    existing = [str(p) for p in nvidia_lib_dirs if p.exists()]
    if existing:
        env["LD_LIBRARY_PATH"] = ":".join(existing + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def append_log(state: SessionState, text: str) -> None:
    state.log = (state.log + "\n\n" + text).strip()


def normalize_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def default_video_path(video_id: str) -> Path:
    return ROOT / "videos" / f"{video_id}.mp4"


def list_local_videos() -> list[str]:
    videos_dir = ROOT / "videos"
    if not videos_dir.exists():
        return []
    return sorted([p.stem for p in videos_dir.glob("*.mp4")])


def build_state(video_id: str, device: str) -> SessionState:
    vid = video_id.strip()
    vpath = default_video_path(vid)
    model = normalize_path(DEFAULT_MODEL_PATH)
    state = SessionState(
        video_id=vid,
        video_path=str(vpath),
        model_path=str(model),
        device=device,
        server_id=1,
        z_hit=2.775,
        z_bounce=0.0,
        ball_csv=str(ROOT / "output" / "ball" / f"{vid}_predict_ball.csv"),
        line_npy=str(ROOT / "output" / "line" / f"{vid}.npy"),
        pose_dir=str(ROOT / "output" / "pose_keypoints" / vid),
        pose_npz=str(ROOT / "output" / "pose_keypoints" / vid / "dump.npz"),
        players_npy=str(ROOT / "output" / "pose_keypoints" / vid / "2_keypoints.npy"),
        hit_bounce_csv=str(ROOT / "output" / "hit_bounce" / f"{vid}.csv"),
        step1_json=str(ROOT / "output" / "step1_2d" / vid / "step1_2d.json"),
        step2_json=str(ROOT / "output" / "step2_velocity" / vid / "step2_velocity.json"),
        step2_traj_csv=str(ROOT / "output" / "step2_trajectory" / vid / "step2_trajectory.csv"),
        step2_traj_png=str(ROOT / "output" / "step2_trajectory" / vid / "step2_trajectory.png"),
        current_stage=1,
        completed={key: False for key in STAGE_KEYS},
    )
    state.completed["prepare"] = True
    append_log(state, f"Prepared session for video_id={vid}\nvideo={state.video_path}")
    return state


def require_state(state_dict: Optional[dict[str, Any]]) -> SessionState:
    if not state_dict:
        raise gr.Error("请先点击“开始分析”")
    return SessionState(**state_dict)


def run_subprocess(stage_name: str, cmd: list[str], device: str) -> str:
    env = build_gpu_env() if device == "cuda" else os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    combined = f"[STAGE] {stage_name}\n[CMD] {' '.join(cmd)}\n"
    if proc.stdout.strip():
        combined += f"\n{proc.stdout.strip()}\n"
    if proc.stderr.strip():
        combined += f"\n[STDERR]\n{proc.stderr.strip()}\n"
    if proc.returncode != 0:
        raise RuntimeError(combined.strip())
    return combined.strip()


def stream_subprocess(state: SessionState, stage_name: str, cmd: list[str]):
    env = build_gpu_env() if state.device == "cuda" else os.environ.copy()
    header = f"[STAGE] {stage_name}\n[CMD] {' '.join(cmd)}"
    base_log = state.log
    current = header
    state.log = (base_log + "\n\n" + current).strip()
    yield render(state)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        current += f"\n{line}"
        state.log = (base_log + "\n\n" + current).strip()
        yield render(state)

    return_code = proc.wait()
    if return_code != 0:
        raise gr.Error(current)


def read_hit_bounce(csv_path: str) -> tuple[str, str]:
    with Path(csv_path).open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return row["hit"], row["bounce"]


def mark_done(state: SessionState, stage_key: str) -> None:
    state.completed[stage_key] = True
    idx = STAGE_KEYS.index(stage_key)
    state.current_stage = min(len(STAGE_KEYS), idx + 1)


def file_update(path_str: str, visible: bool):
    if visible and Path(path_str).exists():
        return gr.update(value=path_str, visible=True)
    return gr.update(value=None, visible=False)


def stage_track_html(state: SessionState) -> str:
    pills = []
    for idx, title in enumerate(STAGE_TITLES):
        if idx < state.current_stage:
            cls = "done"
        elif idx == state.current_stage:
            cls = "current"
        else:
            cls = "todo"
        pills.append(f"<div class='stage-pill {cls}'>{idx + 1}. {title}</div>")
    return "<div class='stage-track'>" + "".join(pills) + "</div>"


def cta_label(state: Optional[SessionState]) -> str:
    if state is None or not state.video_id:
        return "开始分析"
    return "继续下一步"


def metric_html(label: str, value: str, sub: str) -> str:
    display = value if value else "等待生成"
    return (
        "<div class='result-card'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{display}</div>"
        f"<div class='sub'>{sub}</div>"
        "</div>"
    )


def build_summary(state: SessionState) -> str:
    payload = {
        "video_id": state.video_id,
        "hit": state.last_hit,
        "bounce": state.last_bounce,
        "speed_kmh": state.last_speed_kmh,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def trajectory_placeholder(state: SessionState) -> str:
    if state.completed.get("step2", False) and Path(state.step2_traj_png).exists():
        return ""
    return "轨迹图将在完成“速度求解”后显示。\n\n你现在可以点击主按钮继续推进流程，或在右侧展开单步执行进行调试。"


def render(state: SessionState):
    files = [
        file_update(state.ball_csv, state.completed.get("ball", False)),
        file_update(state.line_npy, state.completed.get("line", False)),
        file_update(state.pose_npz, state.completed.get("pose", False)),
        file_update(state.players_npy, state.completed.get("filter", False)),
        file_update(state.hit_bounce_csv, state.completed.get("hit_bounce", False)),
        file_update(state.step1_json, state.completed.get("step1", False)),
        file_update(state.step2_json, state.completed.get("step2", False)),
        file_update(state.step2_traj_csv, state.completed.get("step2", False)),
        file_update(state.step2_traj_png, state.completed.get("step2", False)),
    ]
    traj_image = (
        gr.update(value=state.step2_traj_png, visible=True)
        if state.completed.get("step2", False) and Path(state.step2_traj_png).exists()
        else gr.update(value=None, visible=False)
    )
    placeholder = gr.update(
        value=f"<div class='placeholder-shell'>{trajectory_placeholder(state).replace(chr(10), '<br>')}</div>",
        visible=not state.completed.get("step2", False),
    )
    downloads_visible = any(state.completed.get(k, False) for k in STAGE_KEYS[1:])
    return [
        asdict(state),
        stage_track_html(state),
        cta_label(state),
        metric_html("Hit Frame", state.last_hit, "识别到击球帧后将在这里显示"),
        metric_html("Bounce Frame", state.last_bounce, "识别到落地帧后将在这里显示"),
        metric_html("Serve Speed", f"{state.last_speed_kmh} km/h" if state.last_speed_kmh else "", "完成速度求解后生成最终球速"),
        build_summary(state),
        state.log,
        gr.update(visible=downloads_visible),
        *files,
        traj_image,
        placeholder,
    ]


def prepare_session(video_id: str, device: str):
    state = build_state(video_id, device)
    return render(state)


def build_stage_command(state: SessionState, stage_key: str) -> tuple[str, list[str]]:
    if stage_key == "ball":
        return "Ball detection", [sys.executable, "predict.py", "--video_path", state.video_path, "--model_path", state.model_path, "--only_csv", "--device", state.device]
    if stage_key == "line":
        return "Court line detection", [sys.executable, "detect_line_for_video.py", "--video_path", state.video_path, "--out_npy", state.line_npy]
    if stage_key == "pose":
        pose_code = (
            "from estimate_pose import dump_pose_from_video; "
            f"dump_pose_from_video(video_path={state.video_path!r}, out_dir={state.pose_dir!r}, device={state.device!r}, backend='onnxruntime', mode='performance', to_openpose=False, max_frames=-1)"
        )
        return "Pose estimation", [sys.executable, "-c", pose_code]
    if stage_key == "filter":
        return "Two-player filter", [sys.executable, "pose_filter.py", "--video_id", state.video_id, "--pose_npz", state.pose_npz, "--line_npy", state.line_npy, "--video_path", state.video_path]
    if stage_key == "hit_bounce":
        return "Hit/Bounce detection", [sys.executable, "hit_bounce.py", "--video", state.video_path, "--ball", state.ball_csv, "--players", state.players_npy, "--out", state.hit_bounce_csv]
    if stage_key == "step1":
        if not state.last_hit or not state.last_bounce:
            state.last_hit, state.last_bounce = read_hit_bounce(state.hit_bounce_csv)
        return "Step1 coordinate mapping", [sys.executable, "step1_standard_2d.py", "--video_id", state.video_id, "--hit_frame", state.last_hit, "--bounce_frame", state.last_bounce, "--server_id", str(state.server_id), "--video_path", state.video_path, "--line_npy", state.line_npy, "--players_npy", state.players_npy, "--ball_csv", state.ball_csv]
    if stage_key == "step2":
        return "Step2 velocity solve", [sys.executable, "step2_initial_velocity.py", "--video_id", state.video_id, "--step1_json", state.step1_json, "--z_hit", str(state.z_hit), "--z_bounce", str(state.z_bounce)]
    raise gr.Error(f"Unsupported stage: {stage_key}")


def finalize_stage(state: SessionState, stage_key: str) -> None:
    if stage_key == "hit_bounce":
        state.last_hit, state.last_bounce = read_hit_bounce(state.hit_bounce_csv)
        append_log(state, f"Parsed hit={state.last_hit}, bounce={state.last_bounce}")
    elif stage_key == "step2":
        with Path(state.step2_json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        state.last_speed_kmh = f"{data['v0_m_s']['speed_kmh']:.2f}"
        append_log(state, f"Computed speed={state.last_speed_kmh} km/h")
    mark_done(state, stage_key)
    if stage_key == "step2":
        state.current_stage = len(STAGE_KEYS) - 1


def execute_stage_stream(state: SessionState, stage_key: str):
    stage_name, cmd = build_stage_command(state, stage_key)
    yield from stream_subprocess(state, stage_name, cmd)
    finalize_stage(state, stage_key)
    yield render(state)


def run_current_step(state_dict: Optional[dict[str, Any]], video_id: str, device: str):
    if not state_dict:
        yield prepare_session(video_id, device)
        return
    state = require_state(state_dict)
    stage_key = STAGE_KEYS[min(state.current_stage, len(STAGE_KEYS) - 1)]
    if stage_key == "prepare":
        yield render(state)
        return
    yield from execute_stage_stream(state, stage_key)


def run_manual(stage_key: str, state_dict: Optional[dict[str, Any]]):
    state = require_state(state_dict)
    yield from execute_stage_stream(state, stage_key)


with gr.Blocks(title="Serve Speed Demo") as demo:
    gr.Markdown(
        """
        <div id="hero">
          <div class="eyebrow">Tennis Serve-Speed</div>
          <p>
            选择本地视频后，按主流程逐步完成检测与速度计算。
            页面优先展示核心操作与最终结果，高级参数、调试日志和输出文件会收纳在次级区域。
          </p>
        </div>
        """
    )

    state = gr.State({})

    with gr.Group(elem_classes=["surface"]):
        with gr.Column(elem_classes=["card-pad-xl"]):
            gr.Markdown("<div class='section-title'>开始分析</div>")
            gr.Markdown("<p class='section-note'>先选择本地视频，再点击主按钮按顺序推进整个分析流程。</p>")

            with gr.Group(elem_classes=["subpanel"]):
                with gr.Column(elem_classes=["card-pad-lg"]):
                    video_id = gr.Dropdown(label="本地视频", choices=list_local_videos(), value="001")

            gr.Markdown("<div class='cta-helper'>主流程会按 准备会话 → 检测 → 求解速度 的顺序自动推进</div>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("")
                with gr.Column(scale=2):
                    main_action_btn = gr.Button(
                        "开始分析",
                        variant="primary",
                        elem_classes=["primary-cta"],
                    )
                with gr.Column(scale=1):
                    gr.Markdown("")

            step_track = gr.HTML("<div class='section-note' style='text-align:center;'>准备会话后，这里会显示当前阶段与整体进度。</div>")

            with gr.Accordion("高级设置", open=False):
                device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda")
                gr.Markdown("<p class='minor-note'>当前演示版本固定使用默认发球侧与高度参数，无需手动调整。</p>")

    gr.Markdown("")

    with gr.Row():
        hit_card = gr.HTML(metric_html("Hit Frame", "", "识别到击球帧后将在这里显示"))
        bounce_card = gr.HTML(metric_html("Bounce Frame", "", "识别到落地帧后将在这里显示"))
        speed_card = gr.HTML(metric_html("Serve Speed", "", "完成速度求解后生成最终球速"))

    with gr.Row():
        with gr.Column(scale=7):
            with gr.Group(elem_classes=["surface"]):
                with gr.Column(elem_classes=["card-pad-xl"]):
                    gr.Markdown("<div class='section-title'>分析结果</div>")
                    gr.Markdown("<p class='section-note'>这里汇总核心结果。生成完成后会自动刷新最新的命中帧、落地帧与球速信息。</p>")
                    summary = gr.Code(language="json", label="Result Summary")

            with gr.Group(elem_classes=["surface"]):
                with gr.Column(elem_classes=["card-pad-xl"]):
                    gr.Markdown("<div class='section-title'>轨迹图</div>")
                    gr.Markdown("<p class='section-note'>完成速度求解后，会在这里展示轨迹可视化结果。</p>")
                    traj_placeholder = gr.HTML(
                        "<div class='placeholder-shell'>轨迹图将在完成“速度求解”后显示。<br><br>你现在可以点击上方主按钮继续推进流程。</div>"
                    )
                    traj_image = gr.Image(label="Trajectory Plot", visible=False, height=340)

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["surface", "side-surface"]):
                with gr.Column(elem_classes=["card-pad-xl"]):
                    gr.HTML(
                        """
                        <div class="side-tip">
                          <div class="tip-title">附属操作</div>
                          <div class="tip-body">这些区域默认不打扰主流程。只有在调试或排查问题时，再展开查看单步执行、日志与输出文件。</div>
                        </div>
                        """
                    )

                    with gr.Accordion("单步执行", open=False):
                        gr.Markdown("<p class='minor-note'>如果你想逐阶段调试，也可以手动触发每一步。</p>")
                        manual_ball = gr.Button("执行：球检测")
                        manual_line = gr.Button("执行：场地线检测")
                        manual_pose = gr.Button("执行：姿态估计")
                        manual_filter = gr.Button("执行：双人筛选")
                        manual_hb = gr.Button("执行：击球 / 落地")
                        manual_step1 = gr.Button("执行：二维映射")
                        manual_step2 = gr.Button("执行：速度求解")

                    with gr.Accordion("输出文件", open=False, visible=False) as downloads_box:
                        gr.Markdown("<p class='download-note'>结果生成后，这里会按流程顺序开放下载对应文件。</p>")
                        ball_file = gr.File(label="ball.csv", visible=False)
                        line_file = gr.File(label="line.npy", visible=False)
                        pose_file = gr.File(label="dump.npz", visible=False)
                        players_file = gr.File(label="2_keypoints.npy", visible=False)
                        hit_bounce_file = gr.File(label="hit_bounce.csv", visible=False)
                        step1_file = gr.File(label="step1_2d.json", visible=False)
                        step2_json_file = gr.File(label="step2_velocity.json", visible=False)
                        step2_csv_file = gr.File(label="step2_trajectory.csv", visible=False)
                        step2_png_file = gr.File(label="step2_trajectory.png", visible=False)

                    with gr.Accordion("调试日志", open=False):
                        log_box = gr.Textbox(lines=13, label="Log")

    gr.Markdown("<div class='footer-note'>Built with Gradio · 以主流程优先、结果优先的产品化界面展示分析过程</div>")

    outputs = [
        state,
        step_track,
        main_action_btn,
        hit_card,
        bounce_card,
        speed_card,
        summary,
        log_box,
        downloads_box,
        ball_file,
        line_file,
        pose_file,
        players_file,
        hit_bounce_file,
        step1_file,
        step2_json_file,
        step2_csv_file,
        step2_png_file,
        traj_image,
        traj_placeholder,
    ]

    main_action_btn.click(run_current_step, [state, video_id, device], outputs)
    manual_ball.click(lambda s: run_manual("ball", s), state, outputs)
    manual_line.click(lambda s: run_manual("line", s), state, outputs)
    manual_pose.click(lambda s: run_manual("pose", s), state, outputs)
    manual_filter.click(lambda s: run_manual("filter", s), state, outputs)
    manual_hb.click(lambda s: run_manual("hit_bounce", s), state, outputs)
    manual_step1.click(lambda s: run_manual("step1", s), state, outputs)
    manual_step2.click(lambda s: run_manual("step2", s), state, outputs)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, share=True)
