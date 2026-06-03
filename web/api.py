from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from modules.processing_queue import ProcessingQueue
from typing import Any, Dict, List, Optional
import os
import uuid
import shutil
from pydantic import BaseModel
from datetime import date
from pathlib import Path
from urllib.parse import quote

from config import (
    ALIGNMENT_MODEL,
    ALLOWED_EXTENSIONS,
    LLM_MODEL_OPTIONS,
    MAX_FILE_SIZE,
    OUTPUT_FOLDER,
    SEGMENT_FILTER_MODE,
    WHISPER_MODEL_SIZE,
)
from modules.llm_processor import FILTER_MODE_DEFINITIONS
from utils import (
    get_srt_from_ctc_result,
    process_chinese_punctuation,
    seconds_to_hhmmss,
    write_to_csv,
    write_to_srt,
    write_to_txt,
    write_transcript_report_docx,
)

temp_dir = os.environ["GRADIO_TEMP_DIR"]

processing_queue = ProcessingQueue()
files_dict = {}
agent_artifacts: Dict[str, Dict[str, str]] = {}

router = APIRouter(prefix="/api")


class createTransribeTaskBody(BaseModel):
    whisper_model_size: str
    llm_model: str
    prompt: Optional[str] = None
    file_path: str


class AgentPathTaskBody(BaseModel):
    file_path: str
    llm_model: str = LLM_MODEL_OPTIONS[0]["label"]
    prompt: Optional[str] = None
    whisper_model_size: str = WHISPER_MODEL_SIZE
    temperature: float = 1.0
    enable_alignment: bool = True
    max_line_length: int = 32
    segment_filter_mode: str = SEGMENT_FILTER_MODE
    enable_second_pass_review: bool = False


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _save_upload_file(file: UploadFile) -> str:
    today = date.today()
    date_str = date.strftime(today, "%Y/%m/%d")
    file_ext = Path(file.filename or "").suffix
    ext = file_ext.lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "unsupported file extension",
                "extension": ext,
                "allowed_extensions": ALLOWED_EXTENSIONS,
            },
        )

    save_dir = f"{temp_dir}/agent-files/{date_str}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    uuid_ex = str(uuid.uuid1()).replace("-", "")
    safe_ext = file_ext or ".mp4"
    save_path = f"{save_dir}/{uuid_ex}{safe_ext}"
    with file.file as src, open(save_path, "wb+") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())

    file_size = os.path.getsize(save_path)
    if file_size > MAX_FILE_SIZE:
        try:
            os.remove(save_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail={
                "message": "file too large",
                "file_size": file_size,
                "max_file_size": MAX_FILE_SIZE,
            },
        )
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="empty uploaded file")

    return save_path


def _create_processing_task(
    files: List[str],
    llm_model: str,
    prompt: Optional[str],
    whisper_model_size: Optional[str],
    temperature: float,
    enable_alignment: bool,
    max_line_length: int,
    segment_filter_mode: str,
    enable_second_pass_review: bool,
) -> str:
    task_id = f"task_{uuid.uuid4().hex}"
    processing_queue.add_task(
        task_id,
        files,
        llm_model,
        prompt,
        temperature=temperature,
        whisper_model_size=whisper_model_size,
        enable_alignment=enable_alignment,
        max_line_length=max_line_length,
        enable_second_pass_review=enable_second_pass_review,
        segment_filter_mode=segment_filter_mode,
    )
    return task_id


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


def _segment_to_row(filename: str, seg: Dict) -> List[str]:
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", start) or start)
    tags = seg.get("tags", "")
    return [
        filename,
        seconds_to_hhmmss(start),
        seconds_to_hhmmss(end),
        seconds_to_hhmmss(max(0.0, end - start)),
        seg.get("summary", ""),
        ", ".join(tags) if isinstance(tags, list) else str(tags or ""),
        seg.get("relevance_level", ""),
        seg.get("intent", ""),
    ]


def _compact_segment(filename: str, seg: Dict) -> Dict[str, Any]:
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", start) or start)
    return {
        "filename": filename,
        "start": start,
        "end": end,
        "duration": max(0.0, end - start),
        "start_hms": seconds_to_hhmmss(start),
        "end_hms": seconds_to_hhmmss(end),
        "summary": seg.get("summary", ""),
        "tags": seg.get("tags", []),
        "relevance_level": seg.get("relevance_level", ""),
        "intent": seg.get("intent", ""),
    }


def _ensure_agent_artifacts(task_id: str, task_result: Dict) -> Dict[str, str]:
    existing = agent_artifacts.get(task_id)
    if existing and all(os.path.exists(path) for path in existing.values()):
        return existing

    if task_result.get("status") != "completed":
        return {}

    output_dir = os.path.join(OUTPUT_FOLDER, task_id, "agent")
    os.makedirs(output_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}
    display_result: List[List[str]] = []

    for file_result in task_result.get("result") or []:
        filename = file_result.get("filename", "")
        base_filename = os.path.splitext(filename)[0] or "transcript"
        align_result = file_result.get("align_result") or {}
        segments = align_result.get("segments") or []

        raw_text = _segments_to_text(segments, "raw_text")
        corrected_text = _segments_to_text(segments, "corrected_text")
        if align_result.get("language") == "zh":
            raw_text = process_chinese_punctuation(raw_text)
            corrected_text = process_chinese_punctuation(corrected_text)
        has_corrected_text_diff = (
            corrected_text.strip()
            and corrected_text.strip() != raw_text.strip()
        )

        raw_txt_path = write_to_txt(
            raw_text,
            output_dir=output_dir,
            filename=f"{base_filename}.txt",
        )
        artifacts[os.path.basename(raw_txt_path)] = raw_txt_path

        if has_corrected_text_diff:
            corrected_txt_path = write_to_txt(
                corrected_text,
                output_dir=output_dir,
                filename=f"{base_filename}_corrected.txt",
            )
            artifacts[os.path.basename(corrected_txt_path)] = corrected_txt_path

        if task_result.get("enable_alignment"):
            if ALIGNMENT_MODEL == "ctc-forced-aligner":
                srt_path = get_srt_from_ctc_result(
                    align_result,
                    max_line_length=int(task_result.get("max_line_length") or 32),
                    output_dir=output_dir,
                    filename=f"{base_filename}.srt",
                )
            else:
                srt_path = write_to_srt(
                    align_result,
                    max_line_length=int(task_result.get("max_line_length") or 32),
                    output_dir=output_dir,
                    filename=f"{base_filename}.srt",
                )
            artifacts[os.path.basename(srt_path)] = srt_path

            if has_corrected_text_diff:
                corrected_align_result = _copy_align_result_with_text_field(
                    align_result,
                    "corrected_text",
                )
                corrected_srt_path = get_srt_from_ctc_result(
                    corrected_align_result,
                    max_line_length=int(task_result.get("max_line_length") or 32),
                    output_dir=output_dir,
                    filename=f"{base_filename}_corrected.srt",
                )
                artifacts[os.path.basename(corrected_srt_path)] = corrected_srt_path

        for seg in file_result.get("segments") or []:
            display_result.append(_segment_to_row(filename, seg))

    result_csv_path = write_to_csv(
        display_result,
        output_dir=output_dir,
        filename="result.csv",
        header=[
            "文件名",
            "开始时间",
            "结束时间",
            "时长",
            "内容摘要",
            "标签",
            "相关度",
            "意图",
        ],
    )
    artifacts[os.path.basename(result_csv_path)] = result_csv_path

    summary_report_path = write_to_txt(
        task_result.get("summary_report") or "本次任务没有生成总结文稿。",
        output_dir=output_dir,
        filename="summary_report.txt",
    )
    artifacts[os.path.basename(summary_report_path)] = summary_report_path

    transcript_report_path = write_transcript_report_docx(
        [
            file_result.get("transcript_document")
            for file_result in task_result.get("result") or []
            if file_result.get("transcript_document")
        ],
        output_dir=output_dir,
        filename="transcript_report.docx",
    )
    artifacts[os.path.basename(transcript_report_path)] = transcript_report_path

    agent_artifacts[task_id] = artifacts
    return artifacts


def _artifact_links(task_id: str, request: Request, artifacts: Dict[str, str]) -> List[Dict[str, str]]:
    base_url = _base_url(request)
    return [
        {
            "name": name,
            "download_url": (
                f"{base_url}/api/agent/tasks/{task_id}/artifacts/{quote(name)}"
            ),
        }
        for name in sorted(artifacts)
    ]


def _agent_task_snapshot(
    task_id: str,
    request: Request,
    include_result: bool = False,
) -> Dict[str, Any]:
    result = processing_queue.get_result(task_id)
    status = result.get("status", "not_found")
    try:
        progress = float(result.get("progress", 0.0) or 0.0)
    except (TypeError, ValueError):
        progress = 0.0
    if status != "completed":
        progress = min(progress, 0.99)
    snapshot: Dict[str, Any] = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "status_info": result.get("status_info", ""),
    }
    if status == "not_found":
        return snapshot
    if status == "error":
        snapshot["error"] = result.get("error", "")
    if status == "completed":
        file_results = result.get("result") or []
        artifacts = _ensure_agent_artifacts(task_id, result)
        snapshot.update({
            "summary_report": result.get("summary_report", ""),
            "files": [
                {
                    "filename": file_result.get("filename", ""),
                    "retained_segments_count": len(file_result.get("segments") or []),
                    "transcript_summary": (
                        (file_result.get("transcript_document") or {}).get("summary", "")
                    ),
                }
                for file_result in file_results
            ],
            "artifacts": _artifact_links(task_id, request, artifacts),
        })
        if include_result:
            snapshot["segments"] = [
                _compact_segment(file_result.get("filename", ""), seg)
                for file_result in file_results
                for seg in (file_result.get("segments") or [])
            ]
            snapshot["transcript_documents"] = [
                file_result.get("transcript_document")
                for file_result in file_results
                if file_result.get("transcript_document")
            ]
    return snapshot


@router.post("/upload")
def upload(file: UploadFile):
    today = date.today()
    date_str = date.strftime(today, "%Y/%m/%d")
    file_ext = Path(file.filename).suffix
    uuid_ex = str(uuid.uuid1()).replace("-", "")
    save_dir = f"{temp_dir}/files/{date_str}"
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    save_path = f"{save_dir}/{uuid_ex}{file_ext}"
    with file.file as src, open(save_path, "wb+") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    return {"file_path": save_path}


@router.post("/tasks")
def createTranscribeTask(body: createTransribeTaskBody):
    task_id = f"task_{uuid.uuid4().hex}"
    file_path = body.file_path
    if not file_path.startswith(f"{temp_dir}/files"):
        raise HTTPException(status_code=403,
                            detail="file is not permit to visit")
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="file not found")
    print(f"添加任务: {task_id}, 文件路径: {file_path}")
    # 添加到处理队列
    processing_queue.add_task(
        task_id, [file_path], body.llm_model, body.prompt,
        whisper_model_size=body.whisper_model_size,
    )
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
def queryTranscribeTask(task_id: str):
    result = processing_queue.get_result(task_id)
    return result


@router.get("/agent/health")
def agent_health():
    return {
        "status": "ok",
        "service": "PreenCut Agent API",
        "supported_llm_models": [model["label"] for model in LLM_MODEL_OPTIONS],
        "default_llm_model": LLM_MODEL_OPTIONS[0]["label"],
        "default_whisper_model_size": WHISPER_MODEL_SIZE,
        "filter_modes": list(FILTER_MODE_DEFINITIONS.keys()),
        "endpoints": {
            "create_task": "POST /api/agent/tasks",
            "create_task_from_path": "POST /api/agent/tasks/from-path",
            "task_status": "GET /api/agent/tasks/{task_id}",
            "task_result": "GET /api/agent/tasks/{task_id}/result",
            "download_artifact": "GET /api/agent/tasks/{task_id}/artifacts/{artifact_name}",
        },
    }


@router.post("/agent/tasks")
def create_agent_task(
    request: Request,
    file: UploadFile = File(...),
    llm_model: str = Form(LLM_MODEL_OPTIONS[0]["label"]),
    prompt: Optional[str] = Form(None),
    whisper_model_size: str = Form(WHISPER_MODEL_SIZE),
    temperature: float = Form(1.0),
    enable_alignment: bool = Form(True),
    max_line_length: int = Form(32),
    segment_filter_mode: str = Form(SEGMENT_FILTER_MODE),
    enable_second_pass_review: bool = Form(False),
):
    save_path = _save_upload_file(file)
    task_id = _create_processing_task(
        [save_path],
        llm_model,
        prompt,
        whisper_model_size,
        temperature,
        enable_alignment,
        max_line_length,
        segment_filter_mode,
        enable_second_pass_review,
    )
    base_url = _base_url(request)
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "video accepted and queued",
        "poll_url": f"{base_url}/api/agent/tasks/{task_id}",
        "result_url": f"{base_url}/api/agent/tasks/{task_id}/result",
    }


@router.post("/agent/tasks/from-path")
def create_agent_task_from_path(request: Request, body: AgentPathTaskBody):
    file_path = body.file_path
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="file not found")
    task_id = _create_processing_task(
        [file_path],
        body.llm_model,
        body.prompt,
        body.whisper_model_size,
        body.temperature,
        body.enable_alignment,
        body.max_line_length,
        body.segment_filter_mode,
        body.enable_second_pass_review,
    )
    base_url = _base_url(request)
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "local file accepted and queued",
        "poll_url": f"{base_url}/api/agent/tasks/{task_id}",
        "result_url": f"{base_url}/api/agent/tasks/{task_id}/result",
    }


@router.get("/agent/tasks/{task_id}")
def query_agent_task(task_id: str, request: Request):
    return _agent_task_snapshot(task_id, request, include_result=False)


@router.get("/agent/tasks/{task_id}/result")
def query_agent_task_result(task_id: str, request: Request):
    return _agent_task_snapshot(task_id, request, include_result=True)


@router.get("/agent/tasks/{task_id}/artifacts/{artifact_name:path}")
def download_agent_artifact(task_id: str, artifact_name: str):
    task_result = processing_queue.get_result(task_id)
    if task_result.get("status") != "completed":
        raise HTTPException(status_code=409, detail="task is not completed")
    artifacts = _ensure_agent_artifacts(task_id, task_result)
    file_path = artifacts.get(artifact_name)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(file_path, filename=os.path.basename(file_path))
