import os
import uuid
import time
import random
import zipfile
import gradio as gr
from config import (
    LLM_MODEL_OPTIONS,
    ENABLE_ALIGNMENT,
    ENABLE_SECOND_PASS_REVIEW,
    SEGMENT_FILTER_MODE,
)
from config import (
    TEMP_FOLDER,
    OUTPUT_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    WHISPER_MODEL_SIZE,
    MAX_FILE_NUMBERS,
    ALIGNMENT_MODEL,
)
from config import MAX_DURATION_SECONDS
from modules.processing_queue import ProcessingQueue
from modules.video_processor import VideoProcessor
from utils import seconds_to_hhmmss, hhmmss_to_seconds, clear_directory_fast \
    , generate_safe_filename, write_to_srt, write_to_csv, \
    get_srt_from_ctc_result, \
    write_to_txt, write_transcript_report_docx, process_chinese_punctuation
from typing import List, Dict, Tuple, Optional
import subprocess

from web.theme import get_theme_css, get_theme_head
from web.ui_constants import (
    CHECKBOX_CHECKED,
    CHECKBOX_UNCHECKED,
    TASK_SELECT_CHECKED,
    TASK_SELECT_UNCHECKED,
    EMPTY_RESULT_TABLE,
    EMPTY_SEGMENT_SELECTION,
    DEFAULT_ENABLE_ALIGNMENT,
)

# 全局实例
processing_queue = ProcessingQueue()


RESULT_TABLE_HEADERS = [
    "文件名",
    "开始时间",
    "结束时间",
    "时长",
    "内容摘要",
    "标签",
    "相关度",
    "意图",
]

SEGMENT_SELECTION_HEADERS = ["选择"] + RESULT_TABLE_HEADERS


def _segment_to_display_row(filename: str, seg: Dict) -> List[str]:
    return [
        filename,
        f"{seconds_to_hhmmss(seg['start'])}",
        f"{seconds_to_hhmmss(seg['end'])}",
        f"{seconds_to_hhmmss(seg['end'] - seg['start'])}",
        seg.get("summary", ""),
        ", ".join(seg["tags"]) if isinstance(seg.get("tags"), list) else seg.get("tags", ""),
        seg.get("relevance_level", ""),
        seg.get("intent", ""),
    ]


def _segments_to_text(segments: List[Dict], text_field: str) -> str:
    lines = []
    for segment in segments or []:
        text = segment.get(text_field)
        if text is None and text_field != "text":
            text = segment.get("text")
        text = str(text or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _copy_align_result_with_text_field(align_result: Dict, text_field: str) -> Dict:
    copied_result = dict(align_result or {})
    copied_segments = []
    for segment in copied_result.get("segments", []) or []:
        copied_segment = dict(segment)
        copied_segment["text"] = (
            copied_segment.get(text_field)
            or copied_segment.get("text")
            or ""
        )
        copied_segments.append(copied_segment)
    copied_result["segments"] = copied_segments
    return copied_result


FILTER_MODE_CHOICES = [
    ("全部相关提及（含负面）", "all_mentions", False),
    ("仅强相关", "strong_only", False),
    ("强相关 + 弱相关", "strong_and_weak", False),
    ("强相关 + 弱相关 + 仅提及（排除负面）", "non_negative_mentions", False),
]

FILTER_MODE_LABEL_TO_VALUE = {
    label: value for label, value, _ in FILTER_MODE_CHOICES
}
FILTER_MODE_LABEL_TO_SECOND_PASS = {
    label: second_pass for label, _, second_pass in FILTER_MODE_CHOICES
}
FILTER_MODE_VALUE_TO_LABEL = {}
for label, value, second_pass in FILTER_MODE_CHOICES:
    if not second_pass and value not in FILTER_MODE_VALUE_TO_LABEL:
        FILTER_MODE_VALUE_TO_LABEL[value] = label

def _wait_for_media_file_ready(
    file_path: str,
    timeout_seconds: float = 15.0,
    stable_checks: int = 2,
    poll_interval_seconds: float = 0.5,
) -> Optional[float]:
    from utils import get_media_duration

    deadline = time.time() + timeout_seconds
    last_size = -1
    stable_count = 0
    last_duration = None

    while time.time() < deadline:
        if not os.path.exists(file_path):
            time.sleep(poll_interval_seconds)
            continue

        current_size = os.path.getsize(file_path)
        if current_size > 0 and current_size == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = current_size

        duration = get_media_duration(file_path)
        if duration is not None:
            last_duration = duration
            if stable_count >= stable_checks:
                return duration

        time.sleep(poll_interval_seconds)

    return last_duration


def check_uploaded_files(files: List) -> str:
    """Validate uploaded files before the task enters the processing queue."""
    if not files:
        raise gr.Error("???????????")

    if len(files) > MAX_FILE_NUMBERS:
        raise gr.Error(
            f"?????? {MAX_FILE_NUMBERS} ??????? {len(files)} ??"
        )

    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)

        file_size = os.path.getsize(file.name)
        if file_size > MAX_FILE_SIZE:
            raise gr.Error(f"???????? ({file_size} > {MAX_FILE_SIZE})")

        ext = os.path.splitext(filename)[1][1:].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise gr.Error(
                f"????????: {ext}, ????: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        duration = _wait_for_media_file_ready(file.name)
        if duration is None:
            raise gr.Error(
                f"?????????????: {filename}\n"
                "????????? ffmpeg / ??????????????"
            )

        if duration > MAX_DURATION_SECONDS:
            duration_minutes = duration / 60
            max_minutes = MAX_DURATION_SECONDS / 60
            raise gr.Error(
                f"????????: {filename}\n"
                f"????: {duration_minutes:.1f} ??\n"
                f"??????: {max_minutes} ??"
            )

        saved_paths.append(file.name)

    return saved_paths


def process_files(files: List, llm_model: str,
                  temperature: float,
                  prompt: Optional[str] = None,
                  whisper_model_size: Optional[str] = None,
                  enable_alignment=None, max_line_length=32,
                  segment_filter_mode="全部相关提及（含负面）") -> Tuple[
    str, Dict, float, Optional[List]]:
    """处理上传的文件，返回 (task_id, status_display, progress_initial, file_upload_clear)."""
    # 检查上传的文件是否符合要求
    saved_paths = check_uploaded_files(files)

    # 上传去重：同一路径只保留一条（避免重复选择同一文件被处理多次）
    original_count = len(saved_paths)
    seen_paths = set()
    deduped_paths = []
    for p in saved_paths:
        norm = os.path.normpath(os.path.abspath(p))
        if norm not in seen_paths:
            seen_paths.add(norm)
            deduped_paths.append(p)
    saved_paths = deduped_paths
    dup_count = original_count - len(saved_paths)

    if not saved_paths:
        raise gr.Error("去重后没有可处理的文件，请重新选择。")

    # 创建唯一任务ID
    task_id = f"task_{uuid.uuid4().hex}"

    print(f"添加任务: {task_id}, 文件路径: {saved_paths}" + (f", 已忽略 {dup_count} 个重复" if dup_count else ""), flush=True)

    # 添加到处理队列
    if enable_alignment == "开启":
        enable_alignment = True
    else:
        enable_alignment = False
    selected_filter_label = segment_filter_mode
    enable_second_pass_review = FILTER_MODE_LABEL_TO_SECOND_PASS.get(
        selected_filter_label,
        ENABLE_SECOND_PASS_REVIEW,
    )
    segment_filter_mode = FILTER_MODE_LABEL_TO_VALUE.get(
        segment_filter_mode,
        SEGMENT_FILTER_MODE,
    )
    processing_queue.add_task(task_id, saved_paths, llm_model, prompt,
                              temperature,
                              whisper_model_size, enable_alignment,
                              max_line_length,
                              enable_second_pass_review,
                              segment_filter_mode)

    status_msg = f"已加入队列，共 {len(saved_paths)} 个文件，请稍候..."
    if dup_count > 0:
        status_msg = f"已忽略 {dup_count} 个重复文件。{status_msg}"
    # 返回 None 用于清空上传栏，便于用户继续上传下一批
    return task_id, {"task_id": task_id, "status": status_msg}, 0.0, None


def cancel_processing(status_display: Dict) -> Dict:
    """请求取消当前任务。从 status_display 中取 task_id，与界面显示一致。"""
    display = status_display or {}
    tid = (display.get("task_id") or "").strip()
    print(f"[取消] 点击取消处理: status_display keys={list(display.keys())}, task_id={tid!r}", flush=True)
    if not tid:
        print("[取消] 无 task_id，无法取消", flush=True)
        return display
    ok = processing_queue.cancel_task(tid)
    print(f"[取消] cancel_task({tid!r}) -> {ok}", flush=True)
    if ok:
        return {"task_id": tid, "status": "正在取消..."}
    return display


def _status_tag_html(status: str, status_display: str) -> str:
    """根据任务状态返回带艺术风格样式的标签 HTML。"""
    status = (status or "").strip().lower()
    if status == "processing":
        cls = "preencut-tag preencut-tag-processing"
    elif status == "completed":
        cls = "preencut-tag preencut-tag-done"
    elif status in ("queued", "cancelled", "not_found"):
        cls = "preencut-tag preencut-tag-pending"
    else:
        cls = "preencut-tag preencut-tag-error"
    return f'<span class="{cls}">{status_display}</span>'


def get_task_detail_list(selected_task_id: Optional[str] = None) -> Tuple[List[List], List[str]]:
    """获取任务详情表格数据。返回 (rows, task_ids_list)，选择列为正方形复选框 HTML，状态列为标签 HTML。"""
    summary = processing_queue.get_all_tasks_summary()
    rows = []
    task_ids_list = []
    for s in summary:
        tid = s["task_id"]
        sel = TASK_SELECT_CHECKED if tid == selected_task_id else TASK_SELECT_UNCHECKED
        status_html = _status_tag_html(s.get("status", ""), s["status_display"])
        rows.append([sel, s["task_id_short"], status_html, s["files_info"], s["submit_time"]])
        task_ids_list.append(tid)
    if not rows:
        rows = [[TASK_SELECT_UNCHECKED, "-", '<span class="preencut-tag preencut-tag-pending">暂无任务</span>', "-", "-"]]
    return rows, task_ids_list


def on_task_table_select(
    evt: gr.SelectData,
    task_ids_list: List[str],
) -> Tuple[List[List], List[str], str]:
    """点击任务详情表任意单元格：选中该行，返回更新后的表格与选中任务 ID。"""
    row_idx = evt.index[0] if evt else -1
    selected_task_id = task_ids_list[row_idx] if (task_ids_list and 0 <= row_idx < len(task_ids_list)) else ""
    rows, ids = get_task_detail_list(selected_task_id)
    return rows, ids, selected_task_id


def ask_confirm_cancel(selected_task_id: str) -> Tuple[str, str, dict, str]:
    """点击「取消」时：若已选中则弹出确认框，否则在操作反馈中提示。返回 (cancel_feedback, confirm_pending, dialog_visible, dialog_msg)。"""
    if not (selected_task_id or "").strip():
        return "请先点击表格中一行选中要取消的任务。", "", gr.update(), ""
    return "", "cancel", gr.update(visible=True), "确定要取消选中的任务吗？"


def ask_confirm_delete(selected_task_id: str) -> Tuple[str, str, dict, str]:
    """点击「删除」时：若已选中则弹出确认框，否则在操作反馈中提示。"""
    if not (selected_task_id or "").strip():
        return "请先点击表格中一行选中要删除的任务。", "", gr.update(), ""
    return "", "delete", gr.update(visible=True), "确定要删除该条信息吗？"


def do_confirm_action(
    selected_task_id: str,
    confirm_pending: str,
) -> Tuple[List[List], List[str], str, str, str, dict]:
    """弹框内点击「确认」：根据 confirm_pending 执行取消或删除，并关闭弹框。"""
    rows, ids = get_task_detail_list(selected_task_id)
    close_dialog = gr.update(visible=False)
    if confirm_pending == "cancel":
        ok = processing_queue.cancel_task(selected_task_id or "")
        rows, ids = get_task_detail_list(selected_task_id)
        return rows, ids, selected_task_id, "已取消。" if ok else "取消失败。", "", close_dialog
    if confirm_pending == "delete":
        ok = processing_queue.delete_task(selected_task_id or "")
        rows, ids = get_task_detail_list(None)
        return rows, ids, "", "已删除。" if ok else "删除失败。", "", close_dialog
    return rows, ids, selected_task_id, "", "", close_dialog


def close_confirm_dialog() -> Tuple[str, str, dict]:
    """弹框内点击「关闭」：关闭弹框并清空确认状态。"""
    return "", "", gr.update(visible=False)


def load_selected_task_progress(
    selected_task_id: str,
    enable_alignment: str,
    max_line_length: int,
) -> Tuple:
    """根据选中的任务 ID 加载该任务到各 Tab（分析结果、重新分析、剪辑选项、字幕文件），并返回进度文案。"""
    tid = (selected_task_id or "").strip()
    if not tid:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "请先在表格中点击一行选中要查询的任务，再点击「进度查询」。",
            "",
        )
    out = check_status(tid, enable_alignment, max_line_length, tid)
    (
        file_download,
        srt_download,
        status_display,
        result_table,
        segment_selection,
        asr_result,
        progress,
        task_detail_rows,
        task_ids_list,
        _timer,
    ) = out
    progress_lines = [
        f"任务 ID: {tid[-8:]}",
        f"状态: {status_display.get('status', '')}",
        f"进度: {progress * 100:.0f}%",
    ]
    if status_display.get("status_info"):
        progress_lines.append(f"当前步骤: {status_display['status_info']}")
    progress_info = "\n".join(progress_lines)
    return (
        file_download,
        srt_download,
        status_display,
        result_table,
        segment_selection,
        asr_result,
        progress,
        task_detail_rows,
        task_ids_list,
        progress_info,
        tid,
    )


def _check_status_for_timer(
    task_id: str,
    enable_alignment: str,
    max_line_length: int,
    selected_task_id: Optional[str] = None,
) -> Tuple[Dict, List, List, List, List, str, float, List[List], List[str], str, gr.Timer]:
    """供定时器调用：在 check_status 返回值基础上增加 progress_info 供任务详情 Tab 展示。"""
    out = check_status(task_id, enable_alignment, max_line_length, selected_task_id)
    (
        fd, srt_d, status_d, res_tbl, seg_sel, asr, progress,
        task_rows, task_ids, timer,
    ) = out
    progress_lines = []
    if task_id and status_d:
        progress_lines = [
            f"任务 ID: {task_id[-8:]}",
            f"状态: {status_d.get('status', '')}",
            f"进度: {progress * 100:.0f}%",
        ]
        if status_d.get("status_info"):
            progress_lines.append(f"当前步骤: {status_d['status_info']}")
    progress_info = "\n".join(progress_lines) if progress_lines else ""
    return (fd, srt_d, status_d, res_tbl, seg_sel, asr, progress, task_rows, task_ids, progress_info, timer)


def check_status(
    task_id: str,
    enable_alignment: str,
    max_line_length: int,
    selected_task_id: Optional[str] = None,
) -> Tuple[Dict, List, List, List, List, str, float, List[List], List[str], gr.Timer]:
    """检查任务状态，返回 (..., task_detail_rows, task_ids_list, timer)."""
    task_detail_rows, task_ids_list = get_task_detail_list(selected_task_id)
    result = processing_queue.get_result(task_id)
    try:
        progress = float(result.get("progress", 0.0) or 0.0)
    except (TypeError, ValueError):
        progress = 0.0
    if result.get("status") != "completed":
        progress = min(progress, 0.99)

    if result["status"] == "completed":
        # 整理结果以便显示
        task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        display_result = []
        clip_result = []
        asr_result = ''  # 页面显示的语音识别结果
        subtitle_paths = []  # 可下载的字幕文件
        for file_result in result["result"]:
            asr_result += f"FileName：{file_result['filename']}\n=======================\n"
            align_result = file_result['align_result']
            segments = align_result['segments']
            base_filename = file_result['filename'].split('.')[0]
            text = _segments_to_text(segments, "raw_text")
            corrected_text = _segments_to_text(segments, "corrected_text")
            if align_result.get('language') == 'zh':
                text = process_chinese_punctuation(text)
                corrected_text = process_chinese_punctuation(corrected_text)
            has_corrected_text_diff = (
                corrected_text.strip()
                and corrected_text.strip() != text.strip()
            )
            asr_result += text + '\n\n'
            for seg in file_result["segments"]:
                row = _segment_to_display_row(file_result["filename"], seg)
                clip_row = row.copy()
                clip_row.insert(0, CHECKBOX_UNCHECKED)  # 添加选择框
                display_result.append(row)
                clip_result.append(clip_row)

            asr_path = write_to_txt(
                text, output_dir=task_output_dir,
                filename=base_filename + '.txt'
            )
            subtitle_paths.append(asr_path)

            if has_corrected_text_diff:
                corrected_txt_path = write_to_txt(
                    corrected_text, output_dir=task_output_dir,
                    filename=base_filename + '_corrected.txt'
                )
                subtitle_paths.append(corrected_txt_path)

            # 保存当前视/音频的srt字幕文件
            if enable_alignment == "开启":
                if ALIGNMENT_MODEL == 'ctc-forced-aligner':
                    # 使用ctc-forced-aligner生成srt
                    srt_path = get_srt_from_ctc_result(
                        align_result,
                        max_line_length=max_line_length,
                        output_dir=task_output_dir,
                        filename=base_filename + '.srt')
                else:
                    srt_path = write_to_srt(align_result,
                                            max_line_length=max_line_length,
                                            output_dir=task_output_dir,
                                            filename=base_filename + '.srt')
                subtitle_paths.append(srt_path)

                if has_corrected_text_diff:
                    corrected_align_result = _copy_align_result_with_text_field(
                        align_result,
                        "corrected_text",
                    )
                    corrected_srt_path = get_srt_from_ctc_result(
                        corrected_align_result,
                        max_line_length=max_line_length,
                        output_dir=task_output_dir,
                        filename=base_filename + '_corrected.srt',
                    )
                    subtitle_paths.append(corrected_srt_path)

        # 将结果保存到csv文件
        result_path = write_to_csv(
            display_result,
            output_dir=task_output_dir,
            filename="result.csv",
            header=RESULT_TABLE_HEADERS,
        )
        summary_report_path = write_to_txt(
            result.get("summary_report") or "本次任务没有生成总结文稿。",
            output_dir=task_output_dir,
            filename="summary_report.txt",
        )
        transcript_report_path = write_transcript_report_docx(
            [
                file_result.get("transcript_document")
                for file_result in result["result"]
                if file_result.get("transcript_document")
            ],
            output_dir=task_output_dir,
            filename="transcript_report.docx",
        )

        return (
            [result_path, summary_report_path, transcript_report_path],
            subtitle_paths,
            {"task_id": task_id, "status": "处理完成",
             "raw_result": result["result"],
             "result": display_result, },
            display_result,
            clip_result,
            asr_result,
            1.0,
            task_detail_rows,
            task_ids_list,
            gr.Timer(active=False)
        )

    elif result["status"] == "error":
        return (
            [], [],
            {"task_id": task_id,
             "status": f"错误: {result.get('error', '未知错误')}"},
            EMPTY_RESULT_TABLE, EMPTY_SEGMENT_SELECTION, '', progress, task_detail_rows, task_ids_list, gr.Timer(active=False)
        )
    elif result["status"] == "queued":
        return (
            [], [],
            {"task_id": task_id,
             "status": f"排队中, 前面还有{processing_queue.get_queue_size()}个任务"},
            EMPTY_RESULT_TABLE, EMPTY_SEGMENT_SELECTION, '', 0.0, task_detail_rows, task_ids_list, gr.update()
        )
    elif result["status"] == "cancelled":
        return (
            [], [],
            {"task_id": task_id, "status": "已取消"},
            EMPTY_RESULT_TABLE, EMPTY_SEGMENT_SELECTION, '', result.get("progress", 0.0), task_detail_rows, task_ids_list, gr.Timer(active=False)
        )

    if task_id:
        # 已请求取消但当前步骤尚未结束：主状态显示「取消中」
        status_label = "取消中" if result.get("cancel_requested") else "处理中..."
        return (
            [], [],
            {"task_id": task_id, "status": status_label,
             "status_info": result.get("status_info", ""),
             "progress": progress},
            EMPTY_RESULT_TABLE, EMPTY_SEGMENT_SELECTION, '', progress, task_detail_rows, task_ids_list, gr.update()
        )
    else:
        return (
            [], [],
            {"task_id": "", "status": ""},
            EMPTY_RESULT_TABLE, EMPTY_SEGMENT_SELECTION, '', 0.0, task_detail_rows, task_ids_list, gr.update()
        )


def select_clip(segment_selection: List[List], evt: gr.SelectData) -> List[
    List]:
    """选择剪辑片段"""
    selected_row = segment_selection[evt.index[0]]
    # 切换选择状态
    selected_row[0] = CHECKBOX_CHECKED \
        if selected_row[0] == CHECKBOX_UNCHECKED else CHECKBOX_UNCHECKED
    return segment_selection


def select_all_segments(segment_selection: List[List]) -> List[List]:
    """全选所有片段"""
    if not segment_selection:
        return segment_selection
    return [[CHECKBOX_CHECKED] + list(row[1:]) for row in segment_selection]


def deselect_all_segments(segment_selection: List[List]) -> List[List]:
    """取消全选"""
    if not segment_selection:
        return segment_selection
    return [[CHECKBOX_UNCHECKED] + list(row[1:]) for row in segment_selection]


def clip_and_download(status_display: Dict,
                      segment_selection: List[List], download_mode: str) -> str:
    """剪辑并下载选择的片段"""
    if not status_display or "raw_result" not in status_display:
        raise gr.Error("无效的处理结果")

    # 获取任务ID用于创建唯一目录
    task_id = status_display.get("task_id",
                                 f"temp_{int(time.time() * 1000)}_{random.randint(1000, 9999)}")
    task_temp_dir = os.path.join(TEMP_FOLDER, task_id)
    task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)

    if os.path.exists(task_output_dir):
        clear_directory_fast(task_output_dir)
    else:
        os.makedirs(task_output_dir, exist_ok=True)
    if os.path.exists(task_temp_dir):
        clear_directory_fast(task_temp_dir)
    else:
        os.makedirs(task_temp_dir, exist_ok=True)

    # 组织文件分段
    file_segments = {}
    for file_data in status_display["raw_result"]:
        file_segments[file_data["filename"]] = {
            "segments": file_data["segments"],
            "filepath": file_data["filepath"],
            "ext": os.path.splitext(file_data["filepath"])[1]  # 获取原始文件扩展名
        }

    selected_segments = [seg for seg in segment_selection if
                         seg[0] == CHECKBOX_CHECKED]

    # 处理"合并成一个文件"的情况
    if download_mode == "合并成一个文件":
        # 检查所有片段格式是否一致
        formats = set()
        for seg in selected_segments:
            filename = seg[1]
            file_ext = file_segments[filename]['ext']
            formats.add(file_ext.lower())

        if len(formats) > 1:
            raise gr.Error(
                "无法合并: 所选片段包含多种格式: " + ", ".join(formats))

    selected_clips = []
    for seg in selected_segments:
        filename = seg[1]
        start = hhmmss_to_seconds(seg[2])
        end = hhmmss_to_seconds(seg[3])

        # 找到对应的原始分段
        for original_seg in file_segments[filename]['segments']:
            if abs(original_seg["start"] - start) <= 0.5 and abs(
                    original_seg["end"] - end) <= 0.5:
                selected_clips.append({
                    "filename": filename,
                    "start": original_seg["start"],
                    "end": original_seg["end"],
                    "filepath": file_segments[filename]['filepath'],
                    "ext": file_segments[filename]['ext']  # 添加扩展名
                })
                break

    # 按文件分组
    clips_by_file = {}
    for clip in selected_clips:
        if clip["filename"] not in clips_by_file:
            clips_by_file[clip["filename"]] = {
                "filepath": clip["filepath"],
                "ext": clip["ext"],
                "segments": []
            }
        clips_by_file[clip["filename"]]['segments'].append({
            "start": clip["start"],
            "end": clip["end"],
        })

    # 处理每个文件
    output_files = []
    for filename, segments in clips_by_file.items():
        input_path = segments['filepath']
        # 生成安全的目录名(一个文件可能有多个片段，放在以这个文件名为名的目录下)
        safe_filename = generate_safe_filename(filename)
        output_folder = os.path.join(task_output_dir, safe_filename)
        os.makedirs(output_folder, exist_ok=True)
        single_file_clips = VideoProcessor.clip_video(input_path,
                                                      segments['segments'],
                                                      output_folder,
                                                      segments['ext'])
        output_files.extend(single_file_clips)

    # 如果只有一个文件，直接返回
    if len(output_files) == 1:
        return output_files[0]

    # 根据用户选择的模式处理
    if download_mode == "合并成一个文件":
        # 合并多个文件
        ext = clips_by_file[next(iter(clips_by_file))]['ext']  # 获取第一个文件的扩展名
        combined_path = os.path.join(task_output_dir, f"combined_output{ext}")

        # 创建文件列表
        combine_list_path = os.path.join(task_temp_dir, "combine_list.txt")
        with open(combine_list_path, 'w', encoding='utf-8') as f:
            for file in output_files:
                # 使用绝对路径，并将 Windows 路径分隔符转换为正斜杠（FFmpeg 要求）
                abs_file_path = os.path.abspath(file).replace('\\', '/')
                f.write(f"file '{abs_file_path}'\n")

        # 合并视频
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', combine_list_path,
            '-c', 'copy', combined_path
        ]
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                encoding='utf-8', errors='replace'
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg 错误 (返回码: {e.returncode}):\n"
            error_msg += f"命令: {' '.join(cmd)}\n"
            if e.stderr:
                error_msg += f"错误输出: {e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr}\n"
            if e.stdout:
                error_msg += f"标准输出: {e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout}"
            print(error_msg)
            raise gr.Error(f"文件合并失败: {error_msg}")

        return combined_path

    # 打包成zip文件
    else:
        # 创建zip文件
        zip_path = os.path.join(task_output_dir, "clipped_segments.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_files:
                # 在zip文件中使用相对路径
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)

        return zip_path


def start_reanalyze() -> Dict:
    return {
        'status': '请稍候，正在使用新的提示重新分析...',
    }


def reanalyze_with_prompt(task_id: str, reanalyze_llm_model: str,
                          new_prompt: str, temperature: float,
                          segment_filter_mode="全部相关提及（含负面）") -> Tuple[
    Dict, List[List], List[List]]:
    """使用新的提示重新分析"""
    if not task_id:
        raise gr.Error("无效的任务ID")
    task_result = processing_queue.get_result(task_id)
    if not task_result or "result" not in task_result:
        raise gr.Error("没有可以重新分析的内容")

    if not new_prompt:
        raise gr.Error("请输入新的分析提示")

    if not reanalyze_llm_model:
        raise gr.Error("请选择大语言模型")

    try:
        # 使用新提示重新处理
        from modules.llm_processor import LLMProcessor
        selected_filter_label = segment_filter_mode
        enable_second_pass_review = FILTER_MODE_LABEL_TO_SECOND_PASS.get(
            selected_filter_label,
            ENABLE_SECOND_PASS_REVIEW,
        )
        llm = LLMProcessor(
            reanalyze_llm_model,
            temperature,
            enable_second_pass_review,
            FILTER_MODE_LABEL_TO_VALUE.get(segment_filter_mode, SEGMENT_FILTER_MODE),
        )
        updated_results = []

        for file_data in task_result["result"]:
            llm_input = _copy_align_result_with_text_field(
                file_data["align_result"],
                "corrected_text",
            )
            new_segments = llm.segment_video(llm_input, new_prompt)
            updated_results.append({
                "filename": file_data["filename"],
                "filepath": file_data["filepath"],
                "align_result": file_data["align_result"],
                "segments": new_segments
            })

        # 整理结果以便显示
        display_result = []
        clip_result = []
        for file_result in updated_results:
            for seg in file_result["segments"]:
                row = _segment_to_display_row(file_result["filename"], seg)
                clip_row = row.copy()
                clip_row.insert(0, CHECKBOX_UNCHECKED)  # 添加选择框
                display_result.append(row)
                clip_result.append(clip_row)

        return ({
                    "task_id": task_id,
                    "status": "重新分析完成，请在分析结果中查看",
                    "result": display_result,
                    "raw_result": updated_results
                }, display_result, clip_result)

    except Exception as e:
        print(f"重新分析失败: {str(e)}")
        task_result["status"] = "error"
        task_result["status_info"] = f"重新分析失败: {str(e)}"
        return task_result, [], []






def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="PreenCut", theme=gr.themes.Soft(primary_hue="orange"), head=get_theme_head(), css=get_theme_css()) as app:
        gr.Markdown(
            '<h1 class="preencut-title">'
            '<span class="preencut-title-main">赞意AI视频剪辑助手</span>'
            '<span class="preencut-title-sep"> — </span>'
            '<span class="preencut-title-en">Good IDEA-AI Preencut</span>'
            '</h1>'
            '<p class="preencut-subtitle">真  专家  敢 求胜  利他</p>',
            elem_id="preencut-title-block",
        )
        gr.Markdown(
            "上传包含语音的视频/音频文件，AI将自动识别语音内容、智能分段，并允许您输入自然语言进行检索。")

        # # 测试按钮：用于验证刷新页面后是否加载最新前端
        # with gr.Row():
        #     test_refresh_btn = gr.Button("🔄 测试按钮-刷新后可见最新", variant="secondary")
        #     test_refresh_msg = gr.Textbox(label="测试反馈", interactive=False, visible=True)

        # def on_test_click():
        #     return "✅ 已点击，说明前端已是最新（刷新生效）"
        # test_refresh_btn.click(on_test_click, outputs=test_refresh_msg)

        ### 开始处理 ###
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.Files(
                    label="上传视频/音频文件",
                    file_count="multiple"
                )

                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("王炸组合：Gemini-3 + temperature=1 + large-v3-turbo")
                    llm_model = gr.Dropdown(
                        choices=[model['label'] for model in LLM_MODEL_OPTIONS],
                        value="gemini-3", label="大语言模型")
                    temperature = gr.Slider(minimum=0.1, maximum=1.5, step=0.1,
                                            value=1,
                                            label="摘要生成灵活度(temperature)")
                    model_size = gr.Dropdown(
                        choices=["large-v3-turbo", "large-v3", "large-v2", "large", "medium",
                                 "small", "base", "tiny"],
                        value=WHISPER_MODEL_SIZE,
                        label="语音识别模型大小"
                    )
                    alignment = gr.Radio(
                        choices=["开启", "关闭"],
                        label="语音文字对齐(开启后可生成srt字幕文件，同时会增加耗时)",
                        value=DEFAULT_ENABLE_ALIGNMENT
                    )
                    max_line_length = gr.Slider(minimum=1, maximum=50, step=1,
                                                value=32,
                                                label="单条字幕最大长度(仅对中文有效)",
                                                visible=True)
                    segment_filter_mode = gr.Dropdown(
                        choices=[label for label, _, _ in FILTER_MODE_CHOICES],
                        value=FILTER_MODE_VALUE_TO_LABEL.get(
                            SEGMENT_FILTER_MODE,
                            "全部相关提及（含负面）",
                        ),
                        label="结果保留范围",
                    )

                prompt_input = gr.Textbox(
                    label="自定义分析提示 (可选)",
                    value="找出所有关于“合生元”及“合生元派星”的品牌露出和口播片段。必须包含关键词提及的前后完整语境、产品功能深度讲解、成分描述以及画面展示部分。特别指令：对于长段落的产品介绍，必须提取完整的中间讲述过程，严禁只截取开头结尾。执行策略为“宁多勿少”，凡是涉及该品牌或产品的上下文关联内容（包括铺垫和总结），请全部保留，确保内容完整性以供商务核算。",
                    lines=2
                )
                with gr.Row():
                    process_btn = gr.Button("开始处理", variant="primary", elem_id="btn-start-process", elem_classes=["preencut-btn-action"])
                    task_detail_btn = gr.Button("任务详情", variant="primary", elem_id="btn-task-detail", elem_classes=["preencut-btn-action"])

                with gr.Row():
                    status_display = gr.JSON(label="处理状态")
                    task_id = gr.Textbox(visible=False)
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.01,
                    label="处理进度",
                    interactive=False,
                    visible=True,
                )
                # 定时器提前定义，供「进度查询」等事件启动轮询
                timer = gr.Timer(2, active=True)

            with gr.Column(scale=3):
                right_tabs = gr.Tabs(selected=0)
                with right_tabs:
                    with gr.Tab("分析结果"):
                        file_download = gr.File(label="下载分析结果")
                        result_table = gr.Dataframe(
                            headers=RESULT_TABLE_HEADERS,
                            datatype=["str"] * len(RESULT_TABLE_HEADERS),
                            interactive=True,
                            wrap=True,
                            max_height=420,
                            elem_id="preencut-result-table",
                        )

                    with gr.Tab("任务详情", id=1):
                        _task_rows0, _task_ids0 = get_task_detail_list()
                        task_ids_state = gr.State(value=_task_ids0)
                        selected_task_state = gr.State(value="")
                        confirm_pending_state = gr.State(value="")
                        task_detail_table = gr.Dataframe(
                            headers=["选择", "任务ID", "状态", "文件", "提交时间"],
                            datatype=["html", "str", "html", "str", "str"],
                            interactive=False,
                            wrap=True,
                            label="点击一行选中该任务（方框内 ✓ 表示选中），再点击下方按钮执行进度查询、取消或删除。",
                            value=_task_rows0,
                            elem_id="preencut-task-detail-table",
                        )
                        with gr.Row():
                            query_progress_btn = gr.Button("进度查询", variant="primary", elem_classes=["preencut-btn-action"])
                            cancel_confirm_btn = gr.Button("取消", variant="secondary")
                            delete_confirm_btn = gr.Button("删除", variant="secondary")
                        with gr.Column(visible=False) as confirm_dialog_column:
                            confirm_dialog_msg = gr.Markdown("", elem_id="confirm_dialog_msg")
                            with gr.Row():
                                confirm_ok_btn = gr.Button("确认", variant="primary", elem_classes=["preencut-btn-action"])
                                confirm_close_btn = gr.Button("关闭", variant="secondary")
                        progress_info_display = gr.Textbox(
                            label="选中任务进度",
                            lines=6,
                            interactive=False,
                            placeholder="先在表格中点击一行选中任务，再点击「进度查询」查看进度并加载到分析结果/剪辑选项/字幕文件等 Tab。",
                        )
                        cancel_feedback = gr.Textbox(
                            label="操作反馈",
                            interactive=False,
                            visible=True,
                        )
                        task_detail_table.select(
                            on_task_table_select,
                            inputs=[task_ids_state],
                            outputs=[task_detail_table, task_ids_state, selected_task_state],
                        )
                        cancel_confirm_btn.click(
                            ask_confirm_cancel,
                            inputs=[selected_task_state],
                            outputs=[cancel_feedback, confirm_pending_state, confirm_dialog_column, confirm_dialog_msg],
                            queue=False,
                        )
                        delete_confirm_btn.click(
                            ask_confirm_delete,
                            inputs=[selected_task_state],
                            outputs=[cancel_feedback, confirm_pending_state, confirm_dialog_column, confirm_dialog_msg],
                            queue=False,
                        )
                        # 弹框内「确认」：根据当前是取消还是删除执行对应操作并关闭弹框
                        confirm_ok_btn.click(
                            do_confirm_action,
                            inputs=[selected_task_state, confirm_pending_state],
                            outputs=[task_detail_table, task_ids_state, selected_task_state, cancel_feedback, confirm_pending_state, confirm_dialog_column],
                        )
                        confirm_close_btn.click(
                            close_confirm_dialog,
                            inputs=None,
                            outputs=[cancel_feedback, confirm_pending_state, confirm_dialog_column],
                            queue=False,
                        )

                    with gr.Tab("重新分析"):
                        new_prompt = gr.Textbox(
                            label="输入新的分析提示",
                            placeholder="例如：找出所有关于“合生元”及“合生元派星”的品牌露出和口播片段。必须包含关键词提及的前后完整语境、产品功能深度讲解、成分描述以及画面展示部分。",
                            lines=2
                        )
                        reanalyze_llm_model = gr.Dropdown(
                            choices=[model['label'] for model in LLM_MODEL_OPTIONS],
                            value="gemini-3", label="大语言模型")
                        reanlyze_temperature = gr.Slider(minimum=0.1, maximum=1.5,
                                                         step=0.1, value=1,
                                                         label="摘要生成灵活度(temperature)")
                        reanalyze_segment_filter_mode = gr.Dropdown(
                            choices=[label for label, _, _ in FILTER_MODE_CHOICES],
                            value=FILTER_MODE_VALUE_TO_LABEL.get(
                                SEGMENT_FILTER_MODE,
                                "全部相关提及（含负面）",
                            ),
                            label="结果保留范围",
                        )
                        reanalyze_btn = gr.Button("重新分析", variant="secondary")

                    with gr.Tab("剪辑选项"):
                        segment_selection = gr.Dataframe(
                            headers=SEGMENT_SELECTION_HEADERS,
                            datatype='html',
                            interactive=False,
                            wrap=True,
                            type="array",
                            label="选择要保留的片段",
                            max_height=420,
                            elem_id="preencut-segment-table",
                        )
                        with gr.Row():
                            select_all_btn = gr.Button("全选", variant="secondary")
                            deselect_all_btn = gr.Button("取消全选", variant="secondary")
                        segment_selection.select(select_clip,
                                                 inputs=segment_selection,
                                                 outputs=segment_selection)
                        select_all_btn.click(
                            select_all_segments,
                            inputs=[segment_selection],
                            outputs=segment_selection
                        )
                        deselect_all_btn.click(
                            deselect_all_segments,
                            inputs=[segment_selection],
                            outputs=segment_selection
                        )
                        # 添加下载模式选择
                        download_mode = gr.Radio(
                            choices=["打包成zip文件", "合并成一个文件"],
                            label="选择多个文件时的处理方式",
                            value="打包成zip文件"
                        )
                        clip_btn = gr.Button("剪辑", variant="primary", elem_classes=["preencut-btn-action"])
                        download_output = gr.File(label="下载剪辑结果")

                    with gr.Tab("字幕文件"):
                        srt_download = gr.File(label='下载txt/srt文件')
                        asr_result = gr.Text(label="语音识别结果", lines=20,
                                             interactive=True)

        # 进度查询：加载选中任务到各 Tab（需在 srt_download、asr_result 等定义之后绑定）
        query_progress_btn.click(
            load_selected_task_progress,
            inputs=[selected_task_state, alignment, max_line_length],
            outputs=[
                file_download,
                srt_download,
                status_display,
                result_table,
                segment_selection,
                asr_result,
                progress_bar,
                task_detail_table,
                task_ids_state,
                progress_info_display,
                task_id,
            ],
        ).then(
            lambda: gr.Timer(active=True),
            inputs=None,
            outputs=timer,
            show_progress="hidden",
        )

        # 定时器轮询当前任务状态（含任务详情 Tab 的进度文案与选中行）
        timer.tick(
            _check_status_for_timer,
            inputs=[task_id, alignment, max_line_length, selected_task_state],
            outputs=[
                file_download,
                srt_download,
                status_display,
                result_table,
                segment_selection,
                asr_result,
                progress_bar,
                task_detail_table,
                task_ids_state,
                progress_info_display,
                timer,
            ],
        )

        # 事件处理
        process_btn.click(
            process_files,
            inputs=[file_upload, llm_model, temperature, prompt_input,
                    model_size, alignment, max_line_length,
                    segment_filter_mode],
            outputs=[task_id, status_display, progress_bar, file_upload],
        ).then(
            lambda: gr.Timer(active=True),
            inputs=None,
            outputs=timer,
            show_progress="hidden"
        )

        # 任务详情按钮：与右侧「任务详情」Tab 绑定，点击后右侧切换到该 Tab 并展示（选中当前任务行）
        def go_to_task_detail_tab_and_select_current(current_task_id):
            tid = (current_task_id or "").strip() if isinstance(current_task_id, str) else ""
            rows, ids = get_task_detail_list(selected_task_id=tid if tid else None)
            return gr.update(selected=1), rows, ids, tid
        task_detail_btn.click(
            go_to_task_detail_tab_and_select_current,
            inputs=[task_id],
            outputs=[right_tabs, task_detail_table, task_ids_state, selected_task_state],
            queue=False,
        )

        reanalyze_btn.click(
            start_reanalyze,
            inputs=None,
            outputs=status_display,
        ).then(
            reanalyze_with_prompt,
            inputs=[task_id, reanalyze_llm_model, new_prompt,
                    reanlyze_temperature, reanalyze_segment_filter_mode],
            outputs=[status_display, result_table, segment_selection],
            show_progress="hidden"
        )

        clip_btn.click(
            clip_and_download,
            inputs=[status_display, segment_selection, download_mode],
            outputs=download_output
        )

        return app
