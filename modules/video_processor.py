import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TEMP_FOLDER
from typing import List, Dict
from utils import generate_safe_filename

# 剪辑时并行运行的 ffmpeg 进程数，多片段时明显缩短总耗时
CLIP_WORKERS = min(4, (os.cpu_count() or 4))


class VideoProcessor:
    @staticmethod
    def extract_audio_segment(
        audio_path: str, start_sec: float, end_sec: float, output_path: str
    ) -> str:
        """从整段音频中截取 [start_sec, end_sec] 到 output_path（16kHz 单声道，与对齐模型一致）。"""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec), "-to", str(end_sec), "-i", audio_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return output_path

    @staticmethod
    def extract_audio(video_path: str, task_id: str) -> str:
        """从视频中提取音频"""

        task_temp_dir = os.path.join(TEMP_FOLDER, task_id)
        os.makedirs(task_temp_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(task_temp_dir, f"{base_name}.wav")

        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return audio_path

    @staticmethod
    def _clip_one(args):
        """单段剪辑（供并行调用）。args: (input_path, clip_path, start_sec, duration_sec)"""
        input_path, clip_path, start_sec, duration_sec = args
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_sec), '-i', input_path,
            '-t', str(duration_sec),
            '-c:v', 'libx264',
            '-c:a', 'copy',
            '-avoid_negative_ts', 'make_zero',
            clip_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            raise RuntimeError(f"视频剪辑失败: {result.stderr}")
        return clip_path

    @staticmethod
    def clip_video(input_path: str, segments: List[Dict], output_folder: str,
                   ext: str) -> List[str]:
        """根据分段剪辑视频（多片段时并行执行以缩短总耗时）"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        safe_filename = generate_safe_filename(base_name, max_length=100)

        tasks = []
        for i, seg in enumerate(segments):
            clip_path = os.path.join(output_folder,
                                     f"{safe_filename}_clip_{i}{ext}")
            start_sec = float(seg['start'])
            duration_sec = float(seg['end']) - start_sec
            tasks.append((input_path, clip_path, start_sec, duration_sec))

        clip_list = [None] * len(tasks)
        n_workers = min(CLIP_WORKERS, len(tasks))
        if n_workers <= 1:
            for i, t in enumerate(tasks):
                VideoProcessor._clip_one(t)
                clip_list[i] = t[1]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(VideoProcessor._clip_one, t): i
                                for i, t in enumerate(tasks)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    clip_list[idx] = tasks[idx][1]
                    future.result()
        return clip_list
