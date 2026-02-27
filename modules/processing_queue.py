from queue import Queue
from threading import Thread, Lock
import os
import time
import json
import math
from datetime import datetime
from modules.speech_recognizers.speech_recognizer_factory import \
    SpeechRecognizerFactory
from modules.aligners.text_aligner import TextAligner
from modules.llm_processor import LLMProcessor
from modules.video_processor import VideoProcessor
from modules.word_segmenter import WordSegmenter
from config import (
    SPEECH_RECOGNIZER_TYPE,
    POST_ASR_CORRECTION_MAP,
    WHISPER_MODEL_SIZE,
    MAX_SEGMENTS_PER_LLM_CALL,
    MAX_LLM_INPUT_CHARS,
    LLM_CHUNK_OVERLAP_SEGMENTS,
    MERGE_SEGMENT_MAX_CHARS,
    DEDUPE_SEGMENTS_ACROSS_FILES,
)
from typing import List, Dict, Optional
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
                 enable_alignment=False, max_line_length=32):
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
                "cancel_requested": False,
            }
        self.queue.put(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """请求取消任务（排队中或处理中）。返回是否找到并已标记取消。"""
        tid = (task_id or "").strip()
        if not tid:
            print("[取消] cancel_task: task_id 为空", flush=True)
            return False
        with self.lock:
            task_result = self.results.get(tid)
            status = task_result.get("status") if task_result else None
            if task_result and status in ("queued", "processing"):
                task_result["cancel_requested"] = True
                task_result["status_info"] = "已请求取消，当前步骤（如语音识别/对齐）完成后将停止"
                print(f"[取消] 已标记任务 {tid!r} 取消 (status={status})", flush=True)
                return True
        print(f"[取消] 未找到可取消任务: task_id={tid!r}, status={status}", flush=True)
        return False

    def get_queue_size(self) -> int:
        """获取队列中的任务数（不包括正在执行的）"""
        return self.queue.qsize()

    def _process_queue(self):
        """处理队列中的任务"""
        while True:
            task_id = self.queue.get()
            task_result = self.results[task_id]
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
                    llm = LLMProcessor(llm_model, temperature)
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
                        if POST_ASR_CORRECTION_MAP:
                            for segment in result["segments"]:
                                text = segment.get("text", "")
                                for wrong, right in POST_ASR_CORRECTION_MAP.items():
                                    text = text.replace(wrong, right)
                                segment["text"] = text

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
                        task_result["status_info"] = "大模型分段中..."

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

                        # 写入纠错后（且若开启则对齐后）的断句文件，即输入给大模型前的版本
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
                        print("调用大模型进行分段...")
                        llm_inputs = [
                            {key: segment.get(key) for key in ["start", "end", "text"]}
                            for segment in result["segments"]
                        ]

                        all_segments = []

                        # 序列化一次用于估算长度（避免过大的 prompt 导致超时或断开）
                        try:
                            llm_inputs_json = json.dumps(
                                llm_inputs, ensure_ascii=False
                            )
                        except TypeError:
                            # 如果有不可序列化对象，退化为逐块序列化
                            llm_inputs_json = ""

                        need_chunk_by_count = len(llm_inputs) > MAX_SEGMENTS_PER_LLM_CALL
                        need_chunk_by_chars = (
                            len(llm_inputs_json) > MAX_LLM_INPUT_CHARS
                            if llm_inputs_json
                            else False
                        )

                        if not need_chunk_by_count and not need_chunk_by_chars:
                            # 单次调用即可
                            if task_result.get("cancel_requested"):
                                with self.lock:
                                    task_result["status"] = "cancelled"
                                    task_result["status_info"] = "用户取消"
                                break
                            segments = llm.segment_video(llm_inputs, prompt)
                            all_segments.extend(segments)
                            print(f"大模型分段完成，段数: {len(all_segments)}")
                        else:
                            # 按顺序切成若干块，块之间带重叠，避免边界漏段；合并时按时间去重
                            step = max(1, MAX_SEGMENTS_PER_LLM_CALL - LLM_CHUNK_OVERLAP_SEGMENTS)
                            total_segments = len(llm_inputs)
                            estimated_chunks = max(1, math.ceil((total_segments - 1) / step))
                            print(
                                f"大模型输入过大，按分片方式调用（重叠{LLM_CHUNK_OVERLAP_SEGMENTS}条）："
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
                                        task_result["status_info"] = "用户取消"
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
                                    f"大模型分段中(第{chunk_index + 1}/{estimated_chunks}片)..."
                                )
                                task_result["progress"] = base + (
                                    0.4 + 0.55 * (chunk_index + 1) / estimated_chunks
                                ) / num_files
                                print(
                                    f"分片调用大模型：[{start_idx}, {end_idx})，"
                                    f"chunk_segments={len(chunk)}, chunk_chars={len(chunk_json)}"
                                )
                                chunk_segments = llm.segment_video(chunk, prompt)
                                for seg in chunk_segments:
                                    key = _segment_time_key(seg)
                                    if key not in seen_keys:
                                        seen_keys.add(key)
                                        all_segments.append(seg)
                                chunk_index += 1
                                start_idx += step
                                if start_idx >= total_segments:
                                    break

                            all_segments.sort(key=lambda s: (float(s.get("start", 0)), float(s.get("end", 0))))
                            print(f"大模型分片分段完成，总段数: {len(all_segments)}（已去重）")
                        task_result["progress"] = (i + 1) / num_files
                        task_result["status_info"] = f"第{i + 1}个文件处理完成"

                        segments = all_segments
                        # 单文件内按 summary 去重，避免大模型分片/重叠导致重复片段
                        segments, in_file_removed = _dedupe_segments_in_file(segments)
                        if in_file_removed > 0:
                            print(f"单文件片段去重: 移除 {in_file_removed} 条重复片段")

                        # 保存结果
                        file_results.append({
                            "filename": os.path.basename(file_path),
                            "align_result": result,
                            "segments": segments,
                            "filepath": file_path
                        })

                    # 多文件时按 summary 跨文件去重（部分内容重复只保留一条）
                    if DEDUPE_SEGMENTS_ACROSS_FILES and len(file_results) > 1:
                        removed = _dedupe_segments_across_files(file_results)
                        if removed > 0:
                            print(f"跨文件片段去重: 移除 {removed} 条重复片段")

                    # 更新结果（若未被用户取消）
                    if task_result.get("status") != "cancelled":
                        with self.lock:
                            task_result["status"] = "completed"
                            task_result["result"] = file_results
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
