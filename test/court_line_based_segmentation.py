"""单遍采样的球场线稳定性分段（快版）。"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from court_detector import CourtDetector


class CourtLineSegmenter:
    def __init__(
        self,
        sample_frame_step: int = 15,
        match_threshold: float = 0.62,
        score_ratio_threshold: float = 0.55,
        unstable_patience: int = 2,
        min_segment_length: int = 90,
        max_segment_length: int = 1800,
        sim_scale: float = 0.6,
        verbose: bool = False,
    ):
        self.sample_frame_step = sample_frame_step
        self.match_threshold = match_threshold
        self.score_ratio_threshold = score_ratio_threshold
        self.unstable_patience = unstable_patience
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.sim_scale = sim_scale
        self.verbose = verbose
        self.detector = CourtDetector(verbose=0)

    def _detect_court(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float]:
        try:
            _ = self.detector.detect(frame, verbose=0)
            if not self.detector.court_warp_matrix:
                return False, None, 0.0
            mat = self.detector.court_warp_matrix[-1]
            score = float(self.detector.court_score)
            if mat is None or not np.isfinite(score):
                return False, None, 0.0
            return True, mat.astype(np.float32), score
        except Exception:
            return False, None, 0.0

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        d = float(np.max(np.abs(a - b)))
        return float(np.exp(-d / self.sim_scale))

    def segment_video(self, video_path: str, output_dir: Optional[str] = None) -> List[Tuple[int, int]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if self.verbose:
            print(f"[INFO] total_frames={total}, fps={fps:.2f}, sample_step={self.sample_frame_step}")

        segments: List[Tuple[int, int]] = []
        cur_start = -1
        cur_end = -1
        ref_mat = None
        ref_score = 0.0
        unstable_cnt = 0

        frame_id = 0
        while frame_id < total:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % self.sample_frame_step != 0:
                frame_id += 1
                continue

            found, mat, score = self._detect_court(frame)
            if not found:
                if cur_start >= 0:
                    unstable_cnt += 1
                frame_id += 1
                continue

            if cur_start < 0:
                cur_start = frame_id
                cur_end = frame_id
                ref_mat = mat
                ref_score = max(score, 1.0)
                unstable_cnt = 0
                if self.verbose:
                    print(f"[OPEN] start={cur_start}, score={score:.1f}")
                frame_id += 1
                continue

            sim = self._similarity(ref_mat, mat)
            score_ok = score >= ref_score * self.score_ratio_threshold
            stable = (sim >= self.match_threshold) and score_ok
            too_long = (frame_id - cur_start) >= self.max_segment_length

            if stable and not too_long:
                cur_end = frame_id
                # 轻微更新参考，跟踪缓慢镜头漂移
                ref_mat = (0.9 * ref_mat + 0.1 * mat).astype(np.float32)
                ref_score = 0.9 * ref_score + 0.1 * max(score, 1.0)
                unstable_cnt = 0
            else:
                unstable_cnt += 1
                if too_long or unstable_cnt >= self.unstable_patience:
                    seg_len = cur_end - cur_start + 1
                    if seg_len >= self.min_segment_length:
                        segments.append((cur_start, cur_end))
                        if self.verbose:
                            print(f"[CLOSE] [{cur_start},{cur_end}] len={seg_len}")
                    cur_start = frame_id
                    cur_end = frame_id
                    ref_mat = mat
                    ref_score = max(score, 1.0)
                    unstable_cnt = 0
                    if self.verbose:
                        print(f"[REOPEN] start={cur_start}, sim={sim:.3f}, score={score:.1f}")

            frame_id += 1

        cap.release()

        if cur_start >= 0 and cur_end >= cur_start:
            seg_len = cur_end - cur_start + 1
            if seg_len >= self.min_segment_length:
                segments.append((cur_start, cur_end))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "segments.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("segment_id,start_frame,end_frame,duration_sec,duration_frame\n")
                for i, (s, e) in enumerate(segments, 1):
                    f.write(f"{i},{s},{e},{(e-s+1)/fps:.3f},{e-s+1}\n")
            if self.verbose:
                print(f"[OK] saved csv: {csv_path}")

        return segments

    def save_segments(self, video_path: str, segments: List[Tuple[int, int]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        for i, (s, e) in enumerate(segments, 1):
            out_path = os.path.join(output_dir, f"segment_{i:03d}.mp4")
            t0 = s / fps
            dur = max((e - s + 1) / fps, 0.01)
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
                "-i", video_path,
                "-c", "copy",
                out_path,
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception:
                # ffmpeg copy 失败时回退到重编码，保证可用
                fallback = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{t0:.6f}", "-t", f"{dur:.6f}",
                    "-i", video_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac",
                    out_path,
                ]
                subprocess.run(fallback, check=True)


def main():
    parser = argparse.ArgumentParser(description="基于球场线稳定性的快速分段")
    parser.add_argument("video_path", help="输入视频路径")
    parser.add_argument("--output-dir", default="./cuts", help="输出目录")
    parser.add_argument("--sample-step", type=int, default=15, help="采样步长(帧)")
    parser.add_argument("--match-threshold", type=float, default=0.62, help="warp相似度阈值")
    parser.add_argument("--score-ratio", type=float, default=0.55, help="检测分数保留比例")
    parser.add_argument("--unstable-patience", type=int, default=2, help="连续不稳定容忍次数")
    parser.add_argument("--min-segment", type=int, default=90, help="最短片段帧数")
    parser.add_argument("--max-segment", type=int, default=1800, help="最长片段帧数")
    parser.add_argument("--save-clips", action="store_true", help="是否输出切片视频")
    parser.add_argument("--verbose", action="store_true", help="打印详情")
    args = parser.parse_args()

    seg = CourtLineSegmenter(
        sample_frame_step=args.sample_step,
        match_threshold=args.match_threshold,
        score_ratio_threshold=args.score_ratio,
        unstable_patience=args.unstable_patience,
        min_segment_length=args.min_segment,
        max_segment_length=args.max_segment,
        verbose=args.verbose,
    )

    segments = seg.segment_video(args.video_path, output_dir=args.output_dir)
    print(f"[INFO] segments={len(segments)}")
    for i, (s, e) in enumerate(segments[:10], 1):
        print(f"  #{i}: [{s},{e}] len={e-s+1}")
    if len(segments) > 10:
        print("  ...")

    if args.save_clips and segments:
        seg.save_segments(args.video_path, segments, os.path.join(args.output_dir, "clips"))
        print(f"[OK] clips saved -> {os.path.join(args.output_dir, 'clips')}")


if __name__ == "__main__":
    main()
