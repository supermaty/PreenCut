import os
import uuid
import time
import random
import zipfile
import gradio as gr
from config import LLM_MODEL_OPTIONS, ENABLE_ALIGNMENT
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
    write_to_txt, process_chinese_punctuation
from typing import List, Dict, Tuple, Optional
import subprocess

# 全局实例
processing_queue = ProcessingQueue()
CHECKBOX_CHECKED = '<span style="display: flex; width: 16px; height: 16px; border: 2px solid blue; background:#4B6BFB ;font-weight: bold;color:white;align-items:center;justify-content:center">✓</span>'
CHECKBOX_UNCHECKED = '<span style="display: flex; width: 16px; height: 16px; border: 2px solid blue;font-weight: bold;color:white;align-items:center;justify-content:center"></span>'
if ENABLE_ALIGNMENT:
    DEFAULT_ENABLE_ALIGNMENT = '开启'
else:
    DEFAULT_ENABLE_ALIGNMENT = '关闭'


def check_uploaded_files(files: List) -> str:
    """检查上传的文件是否符合要求"""
    if not files:
        raise gr.Error("请上传至少一个文件")

    if len(files) > MAX_FILE_NUMBERS:
        raise gr.Error(
            f"上传的文件数量超过限制 ({len(files)} > {MAX_FILE_NUMBERS})")

    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)

        # 检查文件大小
        file_size = os.path.getsize(file.name)
        if file_size > MAX_FILE_SIZE:
            raise gr.Error(f"文件大小超过限制 ({file_size} > {MAX_FILE_SIZE})")

        # 检查文件格式
        ext = os.path.splitext(filename)[1][1:].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise gr.Error(
                f"不支持的文件格式: {ext}, 仅支持: {', '.join(ALLOWED_EXTENSIONS)}")

        # 检查文件时长
        from utils import get_media_duration
        duration = get_media_duration(file.name)
        if duration is not None:
            if duration > MAX_DURATION_SECONDS:
                duration_minutes = duration / 60
                max_minutes = MAX_DURATION_SECONDS / 60
                raise gr.Error(
                    f"文件时长超过限制: {filename}\n"
                    f"当前时长: {duration_minutes:.1f} 分钟\n"
                    f"最大允许时长: {max_minutes} 分钟"
                )
        # 如果无法获取时长（可能是文件损坏或格式问题），给出警告但不阻止
        elif duration is None:
            print(f"警告: 无法获取文件时长: {filename}")

        saved_paths.append(file.name)

    return saved_paths


def process_files(files: List, llm_model: str,
                  temperature: float,
                  prompt: Optional[str] = None,
                  whisper_model_size: Optional[str] = None,
                  enable_alignment=None, max_line_length=32) -> Tuple[
    str, Dict, float]:
    """处理上传的文件，返回 (task_id, status_display, progress_initial)."""
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
    processing_queue.add_task(task_id, saved_paths, llm_model, prompt,
                              temperature,
                              whisper_model_size, enable_alignment,
                              max_line_length)

    status_msg = f"已加入队列，共 {len(saved_paths)} 个文件，请稍候..."
    if dup_count > 0:
        status_msg = f"已忽略 {dup_count} 个重复文件。{status_msg}"
    return task_id, {"status": status_msg}, 0.0


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


def check_status(task_id: str, enable_alignment: str, max_line_length: int) -> \
        Tuple[Dict, List, List, float, gr.Timer]:
    """检查任务状态，返回 (file_download, srt_download, status_display, result_table, segment_selection, asr_result, progress, timer)."""
    result = processing_queue.get_result(task_id)
    progress = result.get("progress", 0.0)

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
            text = '\n'.join([text['text'] for text in
                              file_result['align_result']['segments']])
            if file_result['align_result'].get('language') == 'zh':
                text = process_chinese_punctuation(text)
            asr_result += text + '\n\n'
            for seg in file_result["segments"]:
                row = [file_result["filename"],
                       f"{seconds_to_hhmmss(seg['start'])}",
                       f"{seconds_to_hhmmss(seg['end'])}",
                       f"{seconds_to_hhmmss(seg['end'] - seg['start'])}",
                       seg["summary"],
                       ", ".join(seg["tags"]) if isinstance(
                           seg["tags"], list) else seg["tags"]]
                clip_row = row.copy()
                clip_row.insert(0, CHECKBOX_UNCHECKED)  # 添加选择框
                display_result.append(row)
                clip_result.append(clip_row)

            asr_path = write_to_txt(
                text, output_dir=task_output_dir,
                filename=file_result['filename'].split('.')[0] + '.txt'
            )
            subtitle_paths.append(asr_path)

            # 保存当前视/音频的srt字幕文件
            if enable_alignment == "开启":
                if ALIGNMENT_MODEL == 'ctc-forced-aligner':
                    # 使用ctc-forced-aligner生成srt
                    srt_path = get_srt_from_ctc_result(
                        file_result['align_result'],
                        max_line_length=max_line_length,
                        output_dir=task_output_dir,
                        filename=file_result['filename'].split('.')[
                                     0] + '.srt')
                else:
                    srt_path = write_to_srt(file_result['align_result'],
                                            max_line_length=max_line_length,
                                            output_dir=task_output_dir,
                                            filename=
                                            file_result['filename'].split('.')[
                                                0] + '.srt')
                subtitle_paths.append(srt_path)

        # 将结果保存到csv文件
        result_path = write_to_csv(display_result, output_dir=task_output_dir,
                                   filename="result.csv")

        return (
            result_path,
            subtitle_paths,
            {"task_id": task_id, "status": "处理完成",
             "raw_result": result["result"],
             "result": display_result, },
            display_result,
            clip_result,
            asr_result,
            1.0,
            gr.Timer(active=False)
        )

    elif result["status"] == "error":
        return (
            [], [],
            {"task_id": task_id,
             "status": f"错误: {result.get('error', '未知错误')}"},
            [], [], '', progress, gr.Timer(active=False)
        )
    elif result["status"] == "queued":
        return (
            [], [],
            {"task_id": task_id,
             "status": f"排队中, 前面还有{processing_queue.get_queue_size()}个任务"},
            [], [], '', 0.0, gr.update()
        )
    elif result["status"] == "cancelled":
        return (
            [], [],
            {"task_id": task_id, "status": "已取消"},
            [], [], '', result.get("progress", 0.0), gr.Timer(active=False)
        )

    if task_id:
        # 已请求取消但当前步骤尚未结束：主状态显示「取消中」
        status_label = "取消中" if result.get("cancel_requested") else "处理中..."
        return (
            [], [],
            {"task_id": task_id, "status": status_label,
             "status_info": result.get("status_info", ""),
             "progress": progress},
            [], [], '', progress, gr.update()
        )
    else:
        return (
            [], [],
            {"task_id": "", "status": ""},
            [], [], '', 0.0, gr.update()
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
                          new_prompt: str, temperature: float) -> Tuple[
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
        llm = LLMProcessor(reanalyze_llm_model, temperature)
        updated_results = []

        for file_data in task_result["result"]:
            new_segments = llm.segment_video(file_data["align_result"],
                                             new_prompt)
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
                row = [file_result["filename"],
                       f"{seconds_to_hhmmss(seg['start'])}",
                       f"{seconds_to_hhmmss(seg['end'])}",
                       f"{seconds_to_hhmmss(seg['end'] - seg['start'])}",
                       seg["summary"],
                       ", ".join(seg["tags"]) if isinstance(
                           seg["tags"], list) else seg["tags"]]
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


# Tech Assistant 插件：按 https://ai.goodideaggn.com/tech-assistant 集成
# 等 UMD 加载完成后再 init，避免刷新时脚本未就绪导致小机器人不出现
TECH_ASSISTANT_HEAD = """
<script src="https://ai.goodideaggn.com/tech-assistant/tech-assistant.umd.js"></script>
<script>
(function() {
  var appId = "app_6aaf7312ad1d";
  function tryInit() {
    if (window.TechAssistant && typeof window.TechAssistant.init === "function") {
      window.TechAssistant.init({ applicationId: appId });
      return true;
    }
    return false;
  }
  function initWhenReady() {
    if (tryInit()) return;
    var attempts = 0, maxAttempts = 25;
    var t = setInterval(function() {
      if (tryInit() || ++attempts >= maxAttempts) clearInterval(t);
    }, 200);
  }
  if (document.readyState === "complete") initWhenReady();
  else window.addEventListener("load", initWhenReady);
})();
</script>
"""


def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="PreenCut", theme=gr.themes.Soft(), head=TECH_ASSISTANT_HEAD) as app:
        gr.Markdown("# 🎬 PreenCut-AI视频剪辑助手")
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

                prompt_input = gr.Textbox(
                    label="自定义分析提示 (可选)",
                    value="找出所有关于“合生元”及“合生元派星”的品牌露出和口播片段。必须包含关键词提及的前后完整语境、产品功能深度讲解、成分描述以及画面展示部分。特别指令：对于长段落的产品介绍，必须提取完整的中间讲述过程，严禁只截取开头结尾。执行策略为“宁多勿少”，凡是涉及该品牌或产品的上下文关联内容（包括铺垫和总结），请全部保留，确保内容完整性以供商务核算。",
                    lines=2
                )
                with gr.Row():
                    process_btn = gr.Button("开始处理", variant="primary")
                    cancel_btn = gr.Button("取消处理", variant="secondary")

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

            with gr.Column(scale=3):
                with gr.Tab("分析结果"):
                    file_download = gr.File(label="下载分析结果")
                    result_table = gr.Dataframe(
                        headers=["文件名", "开始时间", "结束时间", "时长",
                                 "内容摘要", "标签"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        interactive=True,
                        wrap=True
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
                    reanalyze_btn = gr.Button("重新分析", variant="secondary")

                with gr.Tab("剪辑选项"):
                    segment_selection = gr.Dataframe(
                        headers=["选择", "文件名", "开始时间", "结束时间",
                                 "时长",
                                 "内容摘要", "标签"],
                        datatype='html',
                        interactive=False,
                        wrap=True,
                        type="array",
                        label="选择要保留的片段"
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
                    clip_btn = gr.Button("剪辑", variant="primary")
                    download_output = gr.File(label="下载剪辑结果")

                with gr.Tab("字幕文件"):
                    srt_download = gr.File(label='下载txt/srt文件')
                    asr_result = gr.Text(label="语音识别结果", lines=20,
                                         interactive=True)

        # 定时器，用于轮询状态
        timer = gr.Timer(2, active=False)
        timer.tick(
            check_status,
            inputs=[task_id, alignment, max_line_length],
            outputs=[
                file_download,
                srt_download,
                status_display,
                result_table,
                segment_selection,
                asr_result,
                progress_bar,
                timer,
            ],
        )

        # 事件处理
        process_btn.click(
            process_files,
            inputs=[file_upload, llm_model, temperature, prompt_input,
                    model_size, alignment, max_line_length],
            outputs=[task_id, status_display, progress_bar],
        ).then(
            lambda: gr.Timer(active=True),
            inputs=None,
            outputs=timer,
            show_progress="hidden"
        )

        # queue=False：取消需立即执行，不能等长任务跑完才轮到
        cancel_btn.click(
            cancel_processing,
            inputs=[status_display],
            outputs=[status_display],
            queue=False,
        )

        reanalyze_btn.click(
            start_reanalyze,
            inputs=None,
            outputs=status_display,
        ).then(
            reanalyze_with_prompt,
            inputs=[task_id, reanalyze_llm_model, new_prompt,
                    reanlyze_temperature],
            outputs=[status_display, result_table, segment_selection],
            show_progress="hidden"
        )

        clip_btn.click(
            clip_and_download,
            inputs=[status_display, segment_selection, download_mode],
            outputs=download_output
        )

        return app
