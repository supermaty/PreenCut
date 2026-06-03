from queue import Queue
from threading import Thread, Lock
import os
import time
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from modules.speech_recognizers.speech_recognizer_factory import \
    SpeechRecognizerFactory
from modules.aligners.text_aligner import TextAligner
from modules.llm_processor import LLMProcessor
from modules.video_processor import VideoProcessor
from modules.word_segmenter import WordSegmenter
from config import (
    SPEECH_RECOGNIZER_TYPE,
    WHISPER_MODEL_SIZE,
    MAX_SEGMENTS_PER_LLM_CALL,
    MAX_LLM_INPUT_CHARS,
    LLM_CHUNK_OVERLAP_SEGMENTS,
    LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS,
    MERGE_SEGMENT_MAX_CHARS,
    DEDUPE_SEGMENTS_ACROSS_FILES,
    POST_ASR_CORRECTION_MAP,
    SEGMENT_RECALL_STRATEGY,
    ANCHOR_RECALL_TERMS,
    ANCHOR_RECALL_PRE_SECONDS,
    ANCHOR_RECALL_POST_SECONDS,
    ANCHOR_RECALL_MERGE_GAP_SECONDS,
    ANCHOR_WINDOW_LLM_CONCURRENCY,
    TRANSCRIPT_LLM_MODEL,
)
from typing import Any, List, Dict, Optional, Sequence
from utils import clear_cache
import re


def _segment_time_key(seg: Dict, tolerance: float = 0.5) -> tuple:
    """用于重叠分片合并去重：按 (start, end) 四舍五入到 tolerance 秒作为唯一键。"""
    s = float(seg.get("start", 0))
    e = float(seg.get("end", 0))
    return (round(s / tolerance) * tolerance, round(e / tolerance) * tolerance)


def _normalize_summary_for_dedup(text: str) -> str:
    """归一化 summary 用于跨文件去重比较：去空格、标点，转连续空白为单空格。"""
    if not text:
        return ""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # 去掉标点等，保留中文/英文/数字
    return s.strip()


def _dedupe_segments_in_file(segments: List[Dict]) -> tuple:
    """
    单文件内按 summary 归一化去重：相同或互为子串的 summary 只保留第一次出现的片段。
    返回 (去重后的列表, 移除条数)。
    """
    if not segments:
        return segments, 0
    seen = set()  # 归一化字符串集合
    kept = []
    for seg in segments:
        summary = (seg.get("summary") or "").strip()
        norm = _normalize_summary_for_dedup(summary) or "__empty__"
        if norm in seen:
            continue
        # 与已有项互为子串则视为重复（如「讲解派星」与「讲解派星奶粉」）
        if any((norm in s or s in norm) for s in seen):
            continue
        seen.add(norm)
        kept.append(seg)
    return kept, len(segments) - len(kept)


def _dedupe_segments_across_files(file_results: List[Dict]) -> int:
    """
    按 summary 归一化后跨文件去重：相同或互为子串的 summary 只保留第一次出现的片段。
    返回被移除的重复片段总数。
    """
    if not file_results or len(file_results) < 2:
        return 0
    seen = set()
    removed = 0
    for file_data in file_results:
        segments = file_data.get("segments") or []
        kept = []
        for seg in segments:
            summary = (seg.get("summary") or "").strip()
            norm = _normalize_summary_for_dedup(summary)
            if not norm or norm in seen:
                if norm:
                    removed += 1
                continue
            if any((norm in s or s in norm) for s in seen):
                removed += 1
                continue
            seen.add(norm)
            kept.append(seg)
        file_data["segments"] = kept
    return removed


def _should_split_failed_llm_chunk(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    retryable_or_size_tokens = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "timeout",
        "timed out",
        "unavailable",
        "high demand",
        "overloaded",
        "context",
        "too large",
        "maximum",
    )
    return any(token in text for token in retryable_or_size_tokens)


def _is_size_related_llm_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "context",
            "too large",
            "maximum",
            "max token",
            "token limit",
        )
    )


def _segment_video_with_split_retry(
    llm: LLMProcessor,
    chunk: List[Dict],
    prompt: Optional[str],
    *,
    depth: int = 0,
) -> List[Dict]:
    try:
        return llm.segment_video(chunk, prompt)
    except Exception as exc:
        min_size = max(1, int(LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS))
        if len(chunk) <= min_size or not _should_split_failed_llm_chunk(exc):
            raise

        mid = len(chunk) // 2
        if mid <= 0 or mid >= len(chunk):
            raise

        indent = "  " * depth
        print(
            f"{indent}大模型分片调用失败，自动拆小重试: "
            f"chunk_segments={len(chunk)} -> {mid}+{len(chunk) - mid}; "
            f"error={type(exc).__name__}: {str(exc)[:180]}",
            flush=True,
        )
        left_segments = _segment_video_with_split_retry(
            llm, chunk[:mid], prompt, depth=depth + 1
        )
        right_segments = _segment_video_with_split_retry(
            llm, chunk[mid:], prompt, depth=depth + 1
        )
        return left_segments + right_segments


def _anchor_pattern(anchor_terms: Sequence[str]) -> re.Pattern:
    terms = sorted({str(term).strip() for term in anchor_terms if str(term).strip()}, key=len, reverse=True)
    if not terms:
        terms = ["派星"]
    return re.compile("|".join(re.escape(term) for term in terms))


def _scan_anchor_matches(llm_inputs: List[Dict]) -> List[Dict]:
    pattern = _anchor_pattern(ANCHOR_RECALL_TERMS)
    matches: List[Dict] = []
    for index, segment in enumerate(llm_inputs or []):
        text_fields = {
            "corrected_text": str(segment.get("corrected_text") or ""),
            "raw_text": str(segment.get("raw_text") or ""),
            "text": str(segment.get("text") or ""),
        }
        terms = set()
        fields = set()
        for field_name, value in text_fields.items():
            for match in pattern.finditer(value):
                terms.add(match.group(0))
                fields.add(field_name)
        if not terms:
            continue
        matches.append(
            {
                "anchor_index": len(matches),
                "segment_index": index,
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", segment.get("start", 0.0)) or 0.0),
                "terms": sorted(terms),
                "matched_fields": sorted(fields),
                "text": (
                    segment.get("corrected_text")
                    or segment.get("text")
                    or segment.get("raw_text")
                    or ""
                ),
            }
        )
    return matches


def _raw_anchor_windows(
    matches: List[Dict],
    media_start: float,
    media_end: float,
) -> List[Dict]:
    windows: List[Dict] = []
    for match in matches:
        start = max(media_start, float(match["start"]) - ANCHOR_RECALL_PRE_SECONDS)
        end = min(media_end, float(match["end"]) + ANCHOR_RECALL_POST_SECONDS)
        windows.append(
            {
                "window_index": len(windows),
                "start": start,
                "end": end,
                "anchor_indices": [match["anchor_index"]],
                "anchor_terms": list(match["terms"]),
            }
        )
    return windows


def _merge_anchor_windows(raw_windows: List[Dict]) -> List[Dict]:
    if not raw_windows:
        return []

    ordered = sorted(raw_windows, key=lambda window: (window["start"], window["end"]))
    merged: List[Dict] = []
    current = dict(ordered[0])
    current["anchor_indices"] = list(current.get("anchor_indices", []))
    current["anchor_terms"] = set(current.get("anchor_terms", []))

    for window in ordered[1:]:
        if float(window["start"]) <= float(current["end"]) + ANCHOR_RECALL_MERGE_GAP_SECONDS:
            current["end"] = max(float(current["end"]), float(window["end"]))
            current["anchor_indices"].extend(window.get("anchor_indices", []))
            current["anchor_terms"].update(window.get("anchor_terms", []))
        else:
            current["window_index"] = len(merged)
            current["anchor_terms"] = sorted(current["anchor_terms"])
            merged.append(current)
            current = dict(window)
            current["anchor_indices"] = list(current.get("anchor_indices", []))
            current["anchor_terms"] = set(current.get("anchor_terms", []))

    current["window_index"] = len(merged)
    current["anchor_terms"] = sorted(current["anchor_terms"])
    merged.append(current)
    return merged


def _segments_in_anchor_window(
    llm_inputs: List[Dict],
    start: float,
    end: float,
) -> List[Dict]:
    segments: List[Dict] = []
    for index, segment in enumerate(llm_inputs or []):
        sub_start = float(segment.get("start", 0.0) or 0.0)
        sub_end = float(segment.get("end", sub_start) or sub_start)
        if sub_end < start or sub_start > end:
            continue
        segments.append(
            {
                "segment_index": index,
                "start": sub_start,
                "end": sub_end,
                "text": segment.get("text") or "",
                "raw_text": segment.get("raw_text") or segment.get("text") or "",
                "corrected_text": segment.get("corrected_text")
                or segment.get("text")
                or "",
            }
        )
    return segments


def _build_anchor_windows(llm_inputs: List[Dict]) -> List[Dict]:
    if not llm_inputs:
        return []

    media_start = float(llm_inputs[0].get("start", 0.0) or 0.0)
    media_end = float(llm_inputs[-1].get("end", media_start) or media_start)
    matches = _scan_anchor_matches(llm_inputs)
    raw_windows = _raw_anchor_windows(matches, media_start, media_end)
    merged_windows = _merge_anchor_windows(raw_windows)

    windows: List[Dict] = []
    for window in merged_windows:
        segments = _segments_in_anchor_window(
            llm_inputs, float(window["start"]), float(window["end"])
        )
        if not segments:
            continue
        enriched = dict(window)
        enriched["segments"] = segments
        enriched["source_segment_start_index"] = segments[0]["segment_index"]
        enriched["source_segment_end_index_exclusive"] = (
            segments[-1]["segment_index"] + 1
        )
        windows.append(enriched)

    return windows


def _estimate_anchor_window_prompt_chars(window: Dict, prompt: Optional[str]) -> int:
    payload_segments = []
    for idx, segment in enumerate(window.get("segments") or []):
        text = str(
            segment.get("text")
            or segment.get("corrected_text")
            or segment.get("raw_text")
            or ""
        ).strip()
        if not text:
            continue
        payload_segments.append(
            {
                "idx": idx,
                "segment_index": segment.get("segment_index", idx),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }
        )

    fixed_prompt_budget = 12000
    window_meta_budget = 1000
    return (
        fixed_prompt_budget
        + window_meta_budget
        + len(str(prompt or ""))
        + len(json.dumps(payload_segments, ensure_ascii=False))
    )


def _split_anchor_window(window: Dict) -> Optional[tuple]:
    segments = window.get("segments") or []
    if len(segments) <= 1:
        return None

    mid = len(segments) // 2
    if mid <= 0 or mid >= len(segments):
        return None

    left = dict(window)
    left["segments"] = segments[:mid]
    left["start"] = segments[0]["start"]
    left["end"] = segments[mid - 1]["end"]
    left["source_segment_start_index"] = segments[0].get("segment_index")
    left["source_segment_end_index_exclusive"] = (
        segments[mid - 1].get("segment_index", mid - 1) + 1
    )

    right = dict(window)
    right["segments"] = segments[mid:]
    right["start"] = segments[mid]["start"]
    right["end"] = segments[-1]["end"]
    right["source_segment_start_index"] = segments[mid].get("segment_index")
    right["source_segment_end_index_exclusive"] = (
        segments[-1].get("segment_index", len(segments) - 1) + 1
    )

    return left, right


def _segment_anchor_window_with_retry(
    llm: LLMProcessor,
    window: Dict,
    prompt: Optional[str],
    *,
    depth: int = 0,
) -> List[Dict]:
    segments = window.get("segments") or []
    estimated_prompt_chars = _estimate_anchor_window_prompt_chars(window, prompt)
    if estimated_prompt_chars > MAX_LLM_INPUT_CHARS:
        split = _split_anchor_window(window)
        if split is None:
            raise ValueError(
                "Anchor window exceeds MAX_LLM_INPUT_CHARS but cannot be split: "
                f"window={window.get('window_index')}, segments={len(segments)}, "
                f"estimated_prompt_chars={estimated_prompt_chars}, "
                f"limit={MAX_LLM_INPUT_CHARS}"
            )

        left, right = split
        indent = "  " * depth
        print(
            f"{indent}Anchor window exceeds prompt limit; split before LLM call: "
            f"window={window.get('window_index')}, segments={len(segments)}, "
            f"prompt_chars~{estimated_prompt_chars}, limit={MAX_LLM_INPUT_CHARS}",
            flush=True,
        )
        return (
            _segment_anchor_window_with_retry(llm, left, prompt, depth=depth + 1)
            + _segment_anchor_window_with_retry(llm, right, prompt, depth=depth + 1)
        )

    try:
        return llm.segment_anchor_window_candidates(window, prompt)
    except Exception as exc:
        segments = window.get("segments") or []
        if len(segments) <= max(1, int(LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS)) or not _should_split_failed_llm_chunk(exc):
            raise

        mid = len(segments) // 2
        if mid <= 0 or mid >= len(segments):
            raise

        indent = "  " * depth
        print(
            f"{indent}锚点窗口精细切段失败，自动拆小重试: "
            f"window={window.get('window_index')}, segments={len(segments)} -> {mid}+{len(segments) - mid}; "
            f"error={type(exc).__name__}: {str(exc)[:180]}",
            flush=True,
        )
        left = dict(window)
        left["segments"] = segments[:mid]
        left["start"] = segments[0]["start"]
        left["end"] = segments[mid - 1]["end"]
        right = dict(window)
        right["segments"] = segments[mid:]
        right["start"] = segments[mid]["start"]
        right["end"] = segments[-1]["end"]
        return (
            _segment_anchor_window_with_retry(llm, left, prompt, depth=depth + 1)
            + _segment_anchor_window_with_retry(llm, right, prompt, depth=depth + 1)
        )


def _build_fallback_summary_report(
    file_results: List[Dict],
    error: Optional[Exception] = None,
) -> str:
    lines = ["# 品牌相关内容总结文稿", ""]
    if error is not None:
        lines.extend([
            "LLM 总结文稿生成失败，以下为自动生成的保留片段清单。",
            f"错误信息：{error}",
            "",
        ])
    total = sum(len(file_data.get("segments") or []) for file_data in file_results or [])
    lines.extend([f"保留片段总数：{total}", ""])
    for file_data in file_results or []:
        filename = file_data.get("filename", "")
        segments = file_data.get("segments") or []
        lines.append(f"## {filename}")
        if not segments:
            lines.append("无保留片段。")
            lines.append("")
            continue
        for seg in segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            lines.append(
                "- "
                f"{start:.2f}-{end:.2f}s | "
                f"{seg.get('relevance_level', '')} | "
                f"{seg.get('intent', '')} | "
                f"{seg.get('summary', '')}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_fallback_transcript_document(
    filename: str,
    segments: List[Dict],
    error: Optional[Exception] = None,
) -> Dict[str, str]:
    transcript = " ".join(
        str(
            segment.get("corrected_text")
            or segment.get("raw_text")
            or segment.get("text")
            or ""
        ).strip()
        for segment in segments or []
        if str(
            segment.get("corrected_text")
            or segment.get("raw_text")
            or segment.get("text")
            or ""
        ).strip()
    )
    summary = "主题：婴幼儿奶粉选择\n总结："
    if error is not None:
        summary += f"LLM 文稿修正失败，已使用纠错字幕兜底。错误信息：{error}"
    else:
        summary += "未生成 LLM 总结。"
    return {
        "filename": filename,
        "summary": summary,
        "transcript": transcript,
    }


def _generate_transcript_document_with_fallback(
    llm_model: str,
    temperature: float,
    enable_second_pass_review: bool,
    segment_filter_mode: str,
    filename: str,
    segments: List[Dict],
    prompt: Optional[str],
) -> Dict[str, str]:
    try:
        transcript_llm = LLMProcessor(
            TRANSCRIPT_LLM_MODEL,
            temperature,
            enable_second_pass_review,
            segment_filter_mode,
        )
        print(
            f"Transcript correction/summary LLM model: {TRANSCRIPT_LLM_MODEL}",
            flush=True,
        )
        return transcript_llm.generate_transcript_document(filename, segments, prompt)
    except Exception as transcript_error:
        print(
            f"Transcript report generation failed; using fallback document: {transcript_error}",
            flush=True,
        )
        return _build_fallback_transcript_document(
            filename,
            segments,
            transcript_error,
        )


def _merge_short_segments(segments: List[Dict], max_chars: int) -> List[Dict]:
    """将字数低于 max_chars 的 segment 与上一段合并，减少碎片。"""
    if not segments or max_chars <= 0:
        return segments
    merged: List[Dict] = []
    for seg in segments:
        seg_copy = dict(seg)
        text = (seg_copy.get("text") or "").strip()
        if merged and len(text) < max_chars:
            merged[-1]["end"] = seg_copy.get("end", merged[-1]["end"])
            merged[-1]["text"] = (merged[-1].get("text") or "") + (seg_copy.get("text") or "")
        else:
            merged.append(seg_copy)
    # 若第一条仍过短，合并到第二条
    while len(merged) > 1 and len((merged[0].get("text") or "").strip()) < max_chars:
        merged[1]["start"] = merged[0]["start"]
        merged[1]["text"] = (merged[0].get("text") or "") + (merged[1].get("text") or "")
        merged.pop(0)
    return merged


def _prepare_text_fields_without_local_model(segments: List[Dict]) -> List[Dict]:
    """Keep raw text stable; corrected_text only uses cheap fixed replacements."""
    for segment in segments or []:
        raw_text = str(
            segment.get("raw_text")
            or segment.get("text")
            or ""
        )
        corrected_text = raw_text
        for wrong, right in POST_ASR_CORRECTION_MAP.items():
            corrected_text = corrected_text.replace(wrong, right)
        segment["raw_text"] = raw_text
        segment["text"] = raw_text
        segment["corrected_text"] = corrected_text
    return segments


class ProcessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()
        self.results = {}
        self.result_ttl = 24 * 60 * 60  # 结果保留时间，单位为秒（默认24小时）
        self.max_results = 100  # 最大保留结果数
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
        # 启动清理线程
        self.cleanup_worker = Thread(target=self._cleanup_results, daemon=True)
        self.cleanup_worker.start()

    def add_task(self, task_id: str, files: List[str], llm_model: str,
                 prompt: Optional[str] = None, temperature=0.3,
                 whisper_model_size: Optional[str] = None,
                 enable_alignment=False, max_line_length=32,
                 enable_second_pass_review=False,
                 segment_filter_mode="all_mentions"):
        """添加任务到队列"""
        with self.lock:
            self.results[task_id] = {
                "status": "queued",
                "files": files,
                "prompt": prompt,
                "model_size": whisper_model_size,
                "llm_model": llm_model,
                "timestamp": time.time(),  # 记录任务添加时间
                "temperature": temperature,
                "enable_alignment": enable_alignment,
                "max_line_length": max_line_length,
                "enable_second_pass_review": enable_second_pass_review,
                "segment_filter_mode": segment_filter_mode,
                "cancel_requested": False,
            }
        self.queue.put(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """请求取消任务（排队中或处理中）。返回是否找到并已标记取消。排队任务会立即显示为已取消。"""
        tid = (task_id or "").strip()
        if not tid:
            print("[取消] cancel_task: task_id 为空", flush=True)
            return False
        with self.lock:
            task_result = self.results.get(tid)
            status = task_result.get("status") if task_result else None
            if task_result and status in ("queued", "processing"):
                task_result["cancel_requested"] = True
                task_result["status_info"] = "用户取消"
                # 排队中的任务立即置为已取消，界面可马上显示「已取消」；worker 取到时会跳过执行
                if status == "queued":
                    task_result["status"] = "cancelled"
                else:
                    task_result["status_info"] = "已请求取消，当前步骤（如语音识别/对齐）完成后将停止"
                print(f"[取消] 已标记任务 {tid!r} 取消 (status={status})", flush=True)
                return True
        print(f"[取消] 未找到可取消任务: task_id={tid!r}, status={status}", flush=True)
        return False

    def delete_task(self, task_id: str) -> bool:
        """从结果中移除该任务（仅移除记录，若任务在队列中仍会被 worker 取出后跳过）。返回是否找到并已删除。"""
        tid = (task_id or "").strip()
        if not tid:
            return False
        with self.lock:
            if tid in self.results:
                del self.results[tid]
                print(f"[删除] 已移除任务记录: {tid!r}", flush=True)
                return True
        return False

    def get_task_id_by_suffix(self, suffix: str) -> Optional[str]:
        """根据任务 ID 或后 8 位短 ID 解析出完整 task_id。未找到返回 None。"""
        s = (suffix or "").strip()
        if not s:
            return None
        s_lower = s.lower()
        with self.lock:
            if s in self.results:
                return s
            for tid in self.results:
                if tid.lower().endswith(s_lower) or tid.lower() == s_lower:
                    return tid
        return None

    def get_queue_size(self) -> int:
        """获取队列中的任务数（不包括正在执行的）"""
        return self.queue.qsize()

    def _process_queue(self):
        """处理队列中的任务"""
        while True:
            task_id = self.queue.get()
            task_result = self.results.get(task_id)
            # 任务可能已被删除（delete_task 移除了记录）
            if task_result is None:
                self.queue.task_done()
                continue
            # 若已在取消队列时被置为 cancelled，直接跳过执行并 task_done
            with self.lock:
                if task_result.get("cancel_requested") or task_result.get("status") == "cancelled":
                    task_result["status"] = "cancelled"
                    task_result["status_info"] = "用户取消"
                    print(f"[取消] 任务 {task_id} 已取消（从队列取出时已标记取消）", flush=True)
                    self.queue.task_done()
                    continue
            try:
                with self.lock:
                    task_result["status"] = "processing"
                    task_result["progress"] = 0.0
                    task_result["status_info"] = "准备中..."

                cancelled_early = False
                if task_result.get("cancel_requested"):
                    with self.lock:
                        task_result["status"] = "cancelled"
                        task_result["status_info"] = "用户取消"
                    print(f"[取消] 任务 {task_id} 已取消（准备阶段）", flush=True)
                    cancelled_early = True

                if not cancelled_early:
                    # 获取任务数据
                    with self.lock:
                        files = task_result.get("files")
                        prompt = task_result.get("prompt")
                        model_size = task_result.get("model_size")

                    # 处理每个文件
                    file_results = []
                    llm_model = task_result.get("llm_model")
                    temperature = task_result.get("temperature")
                    llm = LLMProcessor(
                        llm_model,
                        temperature,
                        task_result.get("enable_second_pass_review", False),
                        task_result.get("segment_filter_mode", "all_mentions"),
                    )
                    if task_result['enable_alignment']:
                        word_segmenter = WordSegmenter()

                    num_files = len(files)
                    for i, file_path in enumerate(files):
                        if task_result.get("cancel_requested"):
                            with self.lock:
                                task_result["status"] = "cancelled"
                                task_result["status_info"] = "用户取消"
                            print(f"[取消] 任务 {task_id} 已取消（文件循环内）", flush=True)
                            break
                        base = i / num_files
                        task_result["progress"] = base
                        task_result["status_info"] = f"共{num_files}个文件，正在处理第{i + 1}个文件"
                        # 提取音频（如果是视频）
                        if file_path.lower().endswith(
                                ('.mp4', '.avi', '.mov', '.mkv', '.ts', '.mxf')):
                            audio_path = VideoProcessor.extract_audio(file_path,
                                                                      task_id)
                        else:
                            audio_path = file_path

                        # 语音识别
                        task_result["progress"] = base + 0.05 / num_files
                        task_result["status_info"] = "语音识别中..."
                        print(f"开始语音识别: {file_path}")
                        recognizer = SpeechRecognizerFactory.get_speech_recognizer_by_type(
                            SPEECH_RECOGNIZER_TYPE, model_size)
                        result = recognizer.transcribe(audio_path)
                        print(
                            f"语音识别完成，segments个数: {len(result['segments'])}")
                        del recognizer
                        clear_cache()
                        if task_result.get("cancel_requested"):
                            with self.lock:
                                task_result["status"] = "cancelled"
                                task_result["status_info"] = "用户取消"
                            break
                        task_result["progress"] = base + 0.25 / num_files
                        task_result["status_info"] = "语音识别完成"

                        # ASR 后同音字/专有名词纠错（整词替换）
                        for segment in result["segments"]:
                            raw_text = str(segment.get("text") or "")
                            segment["raw_text"] = raw_text
                            segment["text"] = raw_text

                        if task_result['enable_alignment']:
                            # 文本对齐
                            task_result["status_info"] = "文本对齐中..."
                            print("开始文本对齐...")
                            language = result['language']
                            aligner = TextAligner(language, word_segmenter,
                                                  task_result.get("max_line_length",
                                                                  32))
                            result = aligner.align(result["segments"], audio_path)
                            result["language"] = language
                            print("文本对齐完成")
                            del aligner
                            clear_cache()
                        if task_result.get("cancel_requested"):
                            with self.lock:
                                task_result["status"] = "cancelled"
                                task_result["status_info"] = "用户取消"
                            break
                        task_result["progress"] = base + 0.4 / num_files
                        task_result["status_info"] = "整理 ASR 文本字段中..."

                        # 短句合并：字数低于 MERGE_SEGMENT_MAX_CHARS 的与上一段合并
                        if MERGE_SEGMENT_MAX_CHARS > 0:
                            before_merge = len(result["segments"])
                            result["segments"] = _merge_short_segments(
                                result["segments"], MERGE_SEGMENT_MAX_CHARS
                            )
                            print(
                                f"短句合并: {before_merge} -> {len(result['segments'])} 条 "
                                f"(阈值={MERGE_SEGMENT_MAX_CHARS}字)"
                            )

                        # 不再加载本地纠错模型；只保留 raw_text/corrected_text 字段结构。
                        result["segments"] = _prepare_text_fields_without_local_model(
                            result["segments"]
                        )

                        transcript_document = None
                        transcript_holder: Dict[str, Any] = {}
                        if task_result.get("cancel_requested"):
                            with self.lock:
                                task_result["status"] = "cancelled"
                                task_result["status_info"] = "User cancelled"
                            break
                        task_result["status_info"] = "Starting transcript report and clipping in parallel..."
                        transcript_segments = [
                            dict(segment) for segment in result["segments"]
                        ]

                        def _run_transcript_document_task():
                            transcript_holder["document"] = (
                                _generate_transcript_document_with_fallback(
                                    llm_model,
                                    temperature,
                                    task_result.get("enable_second_pass_review", False),
                                    task_result.get("segment_filter_mode", "all_mentions"),
                                    os.path.basename(file_path),
                                    transcript_segments,
                                    prompt,
                                )
                            )

                        transcript_thread = Thread(
                            target=_run_transcript_document_task,
                            daemon=True,
                        )
                        transcript_thread.start()
                        print(
                            "Transcript report task started in parallel with clipping",
                            flush=True,
                        )
                        segment_data_dir = os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), 'segment-data'
                        )
                        os.makedirs(segment_data_dir, exist_ok=True)
                        seg_model = model_size or WHISPER_MODEL_SIZE
                        segment_list_path = os.path.join(
                            segment_data_dir,
                            f'segment_list_{seg_model}_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
                        )
                        with open(segment_list_path, 'w', encoding='utf-8') as f:
                            json.dump(result["segments"], f, ensure_ascii=False, indent=4)
                        print(f"断句文件已保存: {segment_list_path}")

                        # 调用大模型进行分段（根据字幕数量和字符数进行分片处理）
                        task_result["status_info"] = "大模型分段中..."
                        print("调用大模型进行分段...")
                        llm_inputs = [
                            {
                                "start": segment.get("start"),
                                "end": segment.get("end"),
                                "text": (
                                    segment.get("corrected_text")
                                    or segment.get("text")
                                    or ""
                                ),
                                "raw_text": (
                                    segment.get("raw_text")
                                    or segment.get("text")
                                    or ""
                                ),
                                "corrected_text": (
                                    segment.get("corrected_text")
                                    or segment.get("text")
                                    or ""
                                ),
                            }
                            for segment in result["segments"]
                        ]

                        all_segments = []
                        recall_strategy = SEGMENT_RECALL_STRATEGY
                        if recall_strategy == "anchor_window":
                            anchor_windows = _build_anchor_windows(llm_inputs)
                            anchor_debug_path = os.path.join(
                                "segment-data",
                                f"anchor_windows_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                            )
                            with open(anchor_debug_path, "w", encoding="utf-8") as f:
                                json.dump(anchor_windows, f, ensure_ascii=False, indent=4)
                            print(
                                "Anchor recall + local windows:"
                                f"llm_inputs={len(llm_inputs)}, windows={len(anchor_windows)}, "
                                f"pre={ANCHOR_RECALL_PRE_SECONDS}s, post={ANCHOR_RECALL_POST_SECONDS}s, "
                                f"merge_gap={ANCHOR_RECALL_MERGE_GAP_SECONDS}s"
                            )
                            if not anchor_windows:
                                print("No Paixing anchors matched; skip LLM fine segmentation")
                            else:
                                candidate_segments = []
                                seen_candidate_keys = set()
                                total_windows = len(anchor_windows)
                                window_results: List[List[Dict]] = [
                                    [] for _ in anchor_windows
                                ]
                                max_workers = min(
                                    max(1, int(ANCHOR_WINDOW_LLM_CONCURRENCY)),
                                    total_windows,
                                )
                                print(
                                    "Anchor-window LLM concurrency: "
                                    f"{max_workers}/{total_windows}"
                                )

                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    future_to_index = {}
                                    for window_index, window in enumerate(anchor_windows):
                                        if task_result.get("cancel_requested"):
                                            with self.lock:
                                                task_result["status"] = "cancelled"
                                                task_result["status_info"] = "User cancelled"
                                            break

                                        print(
                                            "Submitting anchor window: "
                                            f"window={window_index}, "
                                            f"time={float(window.get('start', 0)):.2f}-{float(window.get('end', 0)):.2f}s, "
                                            f"segments={len(window.get('segments') or [])}, "
                                            f"anchors={window.get('anchor_terms')}"
                                        )
                                        future = executor.submit(
                                            _segment_anchor_window_with_retry,
                                            llm,
                                            window,
                                            prompt,
                                        )
                                        future_to_index[future] = window_index

                                    completed_windows = 0
                                    try:
                                        for future in as_completed(future_to_index):
                                            window_index = future_to_index[future]
                                            window_results[window_index] = future.result()
                                            completed_windows += 1
                                            task_result["status_info"] = (
                                                "Anchor-window fine segmentation "
                                                f"({completed_windows}/{len(future_to_index)})..."
                                            )
                                            window_progress = min(
                                                1.0,
                                                completed_windows
                                                / max(1, len(future_to_index)),
                                            )
                                            task_result["progress"] = base + (
                                                0.4 + 0.55 * window_progress
                                            ) / num_files
                                    except Exception:
                                        for pending in future_to_index:
                                            pending.cancel()
                                        raise

                                if task_result.get("cancel_requested"):
                                    break

                                for window_candidates in window_results:
                                    for seg in window_candidates:
                                        key = _segment_time_key(seg)
                                        if key not in seen_candidate_keys:
                                            seen_candidate_keys.add(key)
                                            candidate_segments.append(seg)

                                print(
                                    f"Anchor-window fine segmentation complete, candidates={len(candidate_segments)}; running second-stage labeling..."
                                )
                                all_segments = llm.label_and_filter_candidates(
                                    candidate_segments, prompt
                                )
                                print(
                                    f"Second-stage labeling complete, kept={len(all_segments)}"
                                )
                        else:
                            # Legacy fallback: full-subtitle chunking + LLM first-pass recall.
                            try:
                                llm_inputs_json = json.dumps(
                                    llm_inputs, ensure_ascii=False
                                )
                            except TypeError:
                                llm_inputs_json = ""

                            need_chunk_by_count = len(llm_inputs) > MAX_SEGMENTS_PER_LLM_CALL
                            need_chunk_by_chars = (
                                len(llm_inputs_json) > MAX_LLM_INPUT_CHARS
                                if llm_inputs_json
                                else False
                            )

                            if not need_chunk_by_count and not need_chunk_by_chars:
                                if task_result.get("cancel_requested"):
                                    with self.lock:
                                        task_result["status"] = "cancelled"
                                        task_result["status_info"] = "User cancelled"
                                    break
                                segments = _segment_video_with_split_retry(
                                    llm, llm_inputs, prompt
                                )
                                all_segments.extend(segments)
                                print(f"LLM segmentation complete, segments={len(all_segments)}")
                            else:
                                step = max(1, MAX_SEGMENTS_PER_LLM_CALL - LLM_CHUNK_OVERLAP_SEGMENTS)
                                total_segments = len(llm_inputs)
                                estimated_chunks = max(1, math.ceil((total_segments - 1) / step))
                                print(
                                    f"Legacy LLM input too large; chunking with overlap={LLM_CHUNK_OVERLAP_SEGMENTS}: "
                                    f"segments={len(llm_inputs)}, "
                                    f"MAX_SEGMENTS_PER_LLM_CALL={MAX_SEGMENTS_PER_LLM_CALL}, "
                                    f"overlap={LLM_CHUNK_OVERLAP_SEGMENTS}, step={step}"
                                )

                                start_idx = 0
                                chunk_index = 0
                                seen_keys = set()

                                while start_idx < total_segments:
                                    if task_result.get("cancel_requested"):
                                        with self.lock:
                                            task_result["status"] = "cancelled"
                                            task_result["status_info"] = "User cancelled"
                                        break
                                    end_idx = min(
                                        start_idx + MAX_SEGMENTS_PER_LLM_CALL, total_segments
                                    )
                                    chunk = llm_inputs[start_idx:end_idx]

                                    chunk_json = json.dumps(chunk, ensure_ascii=False)
                                    while (
                                        len(chunk_json) > MAX_LLM_INPUT_CHARS
                                        and end_idx - start_idx > 1
                                    ):
                                        end_idx = (start_idx + end_idx) // 2
                                        chunk = llm_inputs[start_idx:end_idx]
                                        chunk_json = json.dumps(chunk, ensure_ascii=False)

                                    task_result["status_info"] = (
                                        f"Legacy LLM segmentation chunk {chunk_index + 1}/{estimated_chunks}..."
                                    )
                                    chunk_progress = min(
                                        1.0,
                                        (chunk_index + 1) / max(1, estimated_chunks),
                                    )
                                    task_result["progress"] = base + (
                                        0.4 + 0.55 * chunk_progress
                                    ) / num_files
                                    print(
                                        f"Calling legacy LLM chunk: [{start_idx}, {end_idx}),"
                                        f"chunk_segments={len(chunk)}, chunk_chars={len(chunk_json)}"
                                    )
                                    chunk_segments = _segment_video_with_split_retry(
                                        llm, chunk, prompt
                                    )
                                    for seg in chunk_segments:
                                        key = _segment_time_key(seg)
                                        if key not in seen_keys:
                                            seen_keys.add(key)
                                            all_segments.append(seg)
                                    chunk_index += 1
                                    processed_count = max(1, end_idx - start_idx)
                                    advance = max(
                                        1,
                                        processed_count - LLM_CHUNK_OVERLAP_SEGMENTS,
                                    )
                                    start_idx += advance
                                    if start_idx >= total_segments:
                                        break

                                all_segments.sort(key=lambda s: (float(s.get("start", 0)), float(s.get("end", 0))))
                                print(f"Legacy chunk segmentation complete, total={len(all_segments)} (deduped)")
                        task_result["progress"] = min(0.96, base + 0.96 / num_files)
                        task_result["status_info"] = f"第{i + 1}个文件剪辑完成，等待文稿任务..."

                        segments = all_segments
                        # 单文件内按 summary 去重，避免大模型分片/重叠导致重复片段
                        segments, in_file_removed = _dedupe_segments_in_file(segments)
                        if in_file_removed > 0:
                            print(f"单文件片段去重: 移除 {in_file_removed} 条重复片段")

                        # 保存结果
                        if transcript_thread.is_alive():
                            task_result["status_info"] = "等待整篇文稿生成..."
                            print("Waiting for transcript report task to finish", flush=True)
                        transcript_thread.join()
                        transcript_document = transcript_holder.get("document")
                        if transcript_document is None:
                            transcript_document = _build_fallback_transcript_document(
                                os.path.basename(file_path),
                                transcript_segments,
                                RuntimeError("Transcript report task did not return a document"),
                            )
                        task_result["progress"] = min(0.97, base + 0.97 / num_files)
                        task_result["status_info"] = f"第{i + 1}个文件处理完成"

                        file_results.append({
                            "filename": os.path.basename(file_path),
                            "align_result": result,
                            "segments": segments,
                            "filepath": file_path,
                            "transcript_document": transcript_document,
                        })

                    # 多文件时按 summary 跨文件去重（部分内容重复只保留一条）
                    if DEDUPE_SEGMENTS_ACROSS_FILES and len(file_results) > 1:
                        removed = _dedupe_segments_across_files(file_results)
                        if removed > 0:
                            print(f"跨文件片段去重: 移除 {removed} 条重复片段")

                    summary_report = ""
                    if task_result.get("status") != "cancelled":
                        task_result["status_info"] = "生成总结文稿..."
                        task_result["progress"] = 0.98
                        try:
                            summary_report = llm.generate_summary_report(
                                file_results,
                                prompt,
                            )
                        except Exception as report_error:
                            print(
                                f"总结文稿生成失败，将使用兜底清单: {report_error}",
                                flush=True,
                            )
                            summary_report = _build_fallback_summary_report(
                                file_results,
                                report_error,
                            )

                    # 更新结果（若未被用户取消）
                    if task_result.get("status") != "cancelled":
                        with self.lock:
                            task_result["status"] = "completed"
                            task_result["result"] = file_results
                            task_result["summary_report"] = summary_report
                            task_result["progress"] = 1.0
                            task_result["status_info"] = "全部完成"

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"任务处理错误: {error_msg}", flush=True)

                with self.lock:
                    task_result["status"] = "error"
                    task_result["error"] = str(e)
            finally:
                self.queue.task_done()

    def get_result(self, task_id: str) -> Dict:
        """获取任务结果"""
        with self.lock:
            result = self.results.get(task_id, {"status": "not_found"})
            if result["status"] != "not_found":
                # 更新访问时间，避免被清理
                result["last_accessed"] = time.time()
            return result

    def get_all_tasks_summary(self) -> List[Dict]:
        """获取所有任务摘要，用于任务详情表格。按提交时间倒序（最近在上）。"""
        status_display_map = {
            "queued": "排队中",
            "processing": "处理中",
            "completed": "已完成",
            "error": "错误",
            "cancelled": "已取消",
            "not_found": "未知",
        }
        with self.lock:
            items = []
            for tid, data in self.results.items():
                status = data.get("status", "not_found")
                status_display = status_display_map.get(status, status)
                files = data.get("files") or []
                if files:
                    files_info = f"{len(files)} 个文件" if len(files) > 1 else os.path.basename(files[0])
                else:
                    files_info = "-"
                ts = data.get("timestamp") or data.get("last_accessed") or 0
                submit_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
                task_id_short = tid[-8:] if len(tid) >= 8 else tid
                items.append({
                    "task_id": tid,
                    "task_id_short": task_id_short,
                    "status": status,
                    "status_display": status_display,
                    "files_info": files_info,
                    "submit_time": submit_time,
                })
            items.sort(key=lambda x: x["submit_time"], reverse=True)
        return items

    def _cleanup_results(self):
        """定期清理过期或过多的结果"""
        while True:
            time.sleep(1 * 60 * 60)  # 每1小时清理一次
            with self.lock:
                current_time = time.time()
                # 按时间排序的结果列表
                sorted_results = sorted(
                    self.results.items(),
                    key=lambda x: x[1].get("last_accessed", x[1]["timestamp"])
                )

                # 移除过期结果
                for task_id, result in list(sorted_results):
                    age = current_time - result.get("last_accessed",
                                                    result["timestamp"])
                    if age > self.result_ttl:
                        self.results.pop(task_id, None)

                # 限制最大结果数
                if len(self.results) > self.max_results:
                    # 删除最旧的结果
                    to_remove = len(self.results) - self.max_results
                    for task_id, _ in sorted_results[:to_remove]:
                        self.results.pop(task_id, None)
