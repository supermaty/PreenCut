import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from config import (
    ENABLE_SECOND_PASS_REVIEW,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MAX_RETRIES,
    LLM_MODEL_OPTIONS,
    LLM_REQUEST_TIMEOUT,
    LLM_RETRY_BASE_SECONDS,
    LLM_RETRY_MAX_SECONDS,
    LLM_TRANSCRIPT_CONCURRENCY,
    MAX_LLM_INPUT_CHARS,
    SEGMENT_FILTER_MODE,
    TRANSCRIPT_LLM_MODEL_OPTIONS,
)

logger = logging.getLogger(__name__)

ALLOWED_RELEVANCE_LEVELS = ("强相关", "弱相关", "仅提及", "负面")
ALLOWED_INTENTS = ("主动讲解", "正面展示", "对比提及", "回答观众", "闲聊提及")
SECOND_PASS_BATCH_SIZE = 20
TARGET_BRAND_NAMES = ("派星", "合生元派星", "合生元")

FILTER_MODE_DEFINITIONS = {
    "all_mentions": {
        "label": "全部相关提及（含负面）",
        "allowed_relevance_levels": {"强相关", "弱相关", "仅提及", "负面"},
    },
    "strong_only": {
        "label": "仅强相关",
        "allowed_relevance_levels": {"强相关"},
    },
    "strong_and_weak": {
        "label": "强相关 + 弱相关",
        "allowed_relevance_levels": {"强相关", "弱相关"},
    },
    "non_negative_mentions": {
        "label": "强相关 + 弱相关 + 仅提及",
        "allowed_relevance_levels": {"强相关", "弱相关", "仅提及"},
    },
}


class LLMProcessor:
    def __init__(
        self,
        llm_model: str,
        temperature: float,
        enable_second_pass_review: Optional[bool] = None,
        segment_filter_mode: Optional[str] = None,
    ):
        for model in [*LLM_MODEL_OPTIONS, *TRANSCRIPT_LLM_MODEL_OPTIONS]:
            if model["label"] == llm_model:
                self.api_key = os.getenv(model["api_key_env_name"])
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=model["base_url"],
                    timeout=float(LLM_REQUEST_TIMEOUT),
                )
                self.model = model["model"]
                self.temperature = temperature
                self.max_tokens = model.get("max_tokens", 4096)
                break

        if not hasattr(self, "client"):
            raise ValueError(
                f"Unsupported LLM model: {llm_model}. Available models: "
                f"{', '.join([m['label'] for m in LLM_MODEL_OPTIONS])}"
            )

        self.enable_second_pass_review = (
            ENABLE_SECOND_PASS_REVIEW
            if enable_second_pass_review is None
            else bool(enable_second_pass_review)
        )
        self.segment_filter_mode = self._normalize_filter_mode(
            segment_filter_mode or SEGMENT_FILTER_MODE
        )

    def _effective_max_tokens(self) -> int:
        return max(1024, min(int(self.max_tokens), int(LLM_MAX_COMPLETION_TOKENS)))

    @staticmethod
    def _is_retryable_llm_error(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            return getattr(exc, "status_code", None) in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            }
        return False

    @staticmethod
    def _error_snippet(exc: Exception) -> str:
        return re.sub(r"\s+", " ", str(exc))[:240]

    def _retry_delay_seconds(self, attempt: int) -> float:
        base_delay = max(0.1, float(LLM_RETRY_BASE_SECONDS))
        max_delay = max(base_delay, float(LLM_RETRY_MAX_SECONDS))
        delay = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0, min(2.0, delay * 0.25))
        return delay + jitter

    def _log_llm_retry(
        self,
        log_label: str,
        exc: Exception,
        attempt: int,
        max_retries: int,
        prompt_chars: int,
        delay: float,
    ) -> None:
        base = str(getattr(self.client, "base_url", None) or "")
        status_code = getattr(exc, "status_code", None)
        message = (
            f"{log_label} transient error attempt {attempt}/{max_retries} "
            f"model={self.model} status={status_code or '-'} "
            f"prompt_chars={prompt_chars}; retrying in {delay:.1f}s. "
            f"{type(exc).__name__}: {self._error_snippet(exc)}"
        )
        print(message, flush=True)
        logger.warning(
            "%s base_url=%s",
            message,
            base[:80] + "..." if len(base) > 80 else base,
        )

    def _normalize_filter_mode(self, value: str) -> str:
        mode = str(value or "").strip()
        if mode in FILTER_MODE_DEFINITIONS:
            return mode
        return "all_mentions"

    def _allowed_relevance_levels(self, filter_mode: Optional[str] = None) -> Set[str]:
        mode = self._normalize_filter_mode(filter_mode or self.segment_filter_mode)
        return set(FILTER_MODE_DEFINITIONS[mode]["allowed_relevance_levels"])

    def _normalize_subtitles(self, subtitles: Any) -> List[Dict]:
        try:
            if isinstance(subtitles, str):
                stripped = subtitles.strip()
                if not stripped:
                    return []
                try:
                    loaded = json.loads(stripped)
                    return self._normalize_subtitles(loaded)
                except Exception:
                    return [{"start": 0.0, "end": 0.0, "text": stripped}]

            if isinstance(subtitles, list):
                normalized: List[Dict] = []
                for item in subtitles:
                    if isinstance(item, dict):
                        start = item.get("start", 0.0) or 0.0
                        end = item.get("end", start)
                        text = str(item.get("text", "") or "").strip()
                        if not text:
                            continue
                        try:
                            start_f = float(start)
                        except (TypeError, ValueError):
                            start_f = 0.0
                        try:
                            end_f = float(end)
                        except (TypeError, ValueError):
                            end_f = start_f
                        normalized.append(
                            {"start": start_f, "end": end_f, "text": text}
                        )
                    else:
                        text = str(item).strip()
                        if text:
                            normalized.append({"start": 0.0, "end": 0.0, "text": text})
                return normalized

            if isinstance(subtitles, dict):
                segments = subtitles.get("segments")
                if isinstance(segments, list):
                    return self._normalize_subtitles(segments)

                text_parts = []
                for value in subtitles.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                text = "\n".join(text_parts) if text_parts else json.dumps(
                    subtitles, ensure_ascii=False
                )
                return [{"start": 0.0, "end": 0.0, "text": text}]
        except Exception as exc:
            logger.exception("Normalize subtitles failed, fallback to string: %s", exc)

        return [{"start": 0.0, "end": 0.0, "text": str(subtitles)}]

    def _strip_code_fence(self, result: Optional[str]) -> str:
        if result is None:
            raise ValueError("LLM returned empty content")
        result = result.strip()
        if result.startswith("```json"):
            return result[7:-3].strip()
        if result.startswith("```"):
            return result[3:-3].strip()
        return result

    def _call_chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        log_label: str,
        temperature: Optional[float] = None,
    ):
        max_retries = max(1, int(LLM_MAX_RETRIES))
        temp = self.temperature if temperature is None else temperature
        prompt_chars = len(system_prompt) + len(user_prompt)
        max_tokens = self._effective_max_tokens()

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                result = self._strip_code_fence(response.choices[0].message.content)
                return json.loads(result)
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                APIStatusError,
            ) as exc:
                if self._is_retryable_llm_error(exc) and attempt < max_retries:
                    delay = self._retry_delay_seconds(attempt)
                    self._log_llm_retry(
                        log_label, exc, attempt, max_retries, prompt_chars, delay
                    )
                    time.sleep(delay)
                    continue
                logger.exception(
                    "%s failed after %d attempt(s) (prompt_chars=%d, max_tokens=%d)",
                    log_label,
                    attempt,
                    prompt_chars,
                    max_tokens,
                )
                raise
            except json.JSONDecodeError:
                logger.exception(
                    "%s returned non-JSON output (prompt_chars=%d)",
                    log_label,
                    prompt_chars,
                )
                raise
            except Exception:
                logger.exception("%s failed (prompt_chars=%d)", log_label, prompt_chars)
                raise

    def _call_chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        log_label: str,
        temperature: Optional[float] = None,
    ) -> str:
        max_retries = max(1, int(LLM_MAX_RETRIES))
        temp = self.temperature if temperature is None else temperature
        prompt_chars = len(system_prompt) + len(user_prompt)
        max_tokens = self._effective_max_tokens()

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                APIStatusError,
            ) as exc:
                if self._is_retryable_llm_error(exc) and attempt < max_retries:
                    delay = self._retry_delay_seconds(attempt)
                    self._log_llm_retry(
                        log_label, exc, attempt, max_retries, prompt_chars, delay
                    )
                    time.sleep(delay)
                    continue
                logger.exception(
                    "%s failed after %d attempt(s) (prompt_chars=%d, max_tokens=%d)",
                    log_label,
                    attempt,
                    prompt_chars,
                    max_tokens,
                )
                raise
            except Exception:
                logger.exception("%s failed (prompt_chars=%d)", log_label, prompt_chars)
                raise

    @staticmethod
    def _format_report_time(seconds: Any) -> str:
        try:
            total = max(0, int(float(seconds)))
        except (TypeError, ValueError):
            total = 0
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _build_report_items(self, file_results: Sequence[Dict]) -> List[Dict]:
        items: List[Dict] = []
        for file_data in file_results or []:
            filename = file_data.get("filename", "")
            for index, seg in enumerate(file_data.get("segments") or [], start=1):
                start = float(seg.get("start", 0.0) or 0.0)
                end = float(seg.get("end", start) or start)
                items.append(
                    {
                        "filename": filename,
                        "index": index,
                        "start": start,
                        "end": end,
                        "time_range": (
                            f"{self._format_report_time(start)}-"
                            f"{self._format_report_time(end)}"
                        ),
                        "duration_seconds": round(max(0.0, end - start), 2),
                        "summary": seg.get("summary", ""),
                        "tags": seg.get("tags", []),
                        "relevance_level": seg.get("relevance_level", ""),
                        "intent": seg.get("intent", ""),
                    }
                )
        return items

    @staticmethod
    def _chunk_report_items(items: List[Dict], max_chars: int) -> List[List[Dict]]:
        chunks: List[List[Dict]] = []
        current: List[Dict] = []
        current_chars = 2
        for item in items:
            item_chars = len(json.dumps(item, ensure_ascii=False)) + 2
            if current and current_chars + item_chars > max_chars:
                chunks.append(current)
                current = []
                current_chars = 2
            current.append(item)
            current_chars += item_chars
        if current:
            chunks.append(current)
        return chunks

    def _generate_report_from_items(
        self,
        items: List[Dict],
        prompt: Optional[str],
        *,
        partial: bool = False,
        chunk_index: int = 1,
        total_chunks: int = 1,
    ) -> str:
        system_prompt = """
你是直播带货品牌内容复盘助手。
你只基于用户提供的已筛选片段生成总结文稿，不得编造片段之外的信息。
输出中文纯文本或 Markdown，不要输出 JSON。
片段引用必须保留文件名与时间段。
""".strip()

        if partial:
            task = (
                f"这是第 {chunk_index}/{total_chunks} 批保留片段。"
                "请生成局部总结，保留可被最终汇总引用的重点。"
            )
        else:
            task = "请生成完整的品牌相关内容总结文稿。"

        user_prompt = f"""
原始筛选要求：
{prompt or "未提供额外筛选要求"}

任务：
{task}

请按以下结构输出：
1. 总体概览：用 3-6 句话概括这批保留片段讲了什么。
2. 相关度分布：按强相关、弱相关、仅提及、负面总结数量和主要内容。
3. 主播表达与卖点：提炼出现过的产品/品牌卖点、场景、话术。
4. 正面可剪内容：列出最值得剪辑的片段，引用格式为「文件名 时间段」。
5. 风险与不适合剪辑内容：列出负面、含糊、仅顺带提及或容易误剪的内容。
6. 详细片段清单：逐条列出保留片段的时间、相关度、意图、摘要。

保留片段 JSON：
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()

        return self._call_chat_text(
            system_prompt,
            user_prompt,
            log_label="LLM summary report",
            temperature=0.2,
        )

    def generate_summary_report(
        self,
        file_results: Sequence[Dict],
        prompt: Optional[str] = None,
    ) -> str:
        items = self._build_report_items(file_results)
        if not items:
            return "本次任务没有筛选出保留片段，因此没有可总结的品牌相关内容。"

        max_chars = max(12000, min(MAX_LLM_INPUT_CHARS // 2, 60000))
        chunks = self._chunk_report_items(items, max_chars)

        if len(chunks) == 1:
            return self._generate_report_from_items(chunks[0], prompt)

        partial_reports = []
        for index, chunk in enumerate(chunks, start=1):
            partial_reports.append(
                self._generate_report_from_items(
                    chunk,
                    prompt,
                    partial=True,
                    chunk_index=index,
                    total_chunks=len(chunks),
                )
            )

        system_prompt = """
你是直播带货品牌内容复盘助手。
你需要把多批局部总结合并成一份完整总结文稿。
只使用局部总结中的信息，不得编造。
输出中文纯文本或 Markdown。
""".strip()
        user_prompt = f"""
原始筛选要求：
{prompt or "未提供额外筛选要求"}

请合并以下局部总结，输出一份完整 summary_report。
结构要求：
1. 总体概览
2. 相关度分布
3. 主播表达与卖点
4. 正面可剪内容
5. 风险与不适合剪辑内容
6. 详细片段清单或重点片段索引

局部总结：
{chr(10).join(f"--- 局部总结 {i + 1} ---{chr(10)}{text}" for i, text in enumerate(partial_reports))}
""".strip()

        return self._call_chat_text(
            system_prompt,
            user_prompt,
            log_label="LLM final summary report",
            temperature=0.2,
        )

    @staticmethod
    def _collapse_transcript_text(text: str) -> str:
        cleaned = str(text or "").replace("#", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _split_text_by_chars(text: str, max_chars: int) -> List[str]:
        text = str(text or "").strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                split_at = max(
                    text.rfind("。", start, end),
                    text.rfind("！", start, end),
                    text.rfind("？", start, end),
                    text.rfind(".", start, end),
                )
                if split_at > start + max_chars * 0.5:
                    end = split_at + 1
            chunks.append(text[start:end].strip())
            start = end
        return [chunk for chunk in chunks if chunk]

    def _run_text_jobs_in_order(
        self,
        count: int,
        call_fn: Callable[[int], str],
        *,
        log_label: str,
    ) -> List[str]:
        if count <= 0:
            return []

        max_workers = min(max(1, int(LLM_TRANSCRIPT_CONCURRENCY)), count)
        if max_workers <= 1:
            return [call_fn(index) for index in range(count)]

        results: List[Optional[str]] = [None] * count
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(call_fn, index): index
                for index in range(count)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    for pending in future_to_index:
                        pending.cancel()
                    logger.exception(
                        "%s concurrent job failed at index=%d/%d",
                        log_label,
                        index + 1,
                        count,
                    )
                    raise

        return [str(result or "") for result in results]

    def _correct_transcript_chunk(self, text: str, chunk_index: int, total_chunks: int) -> str:
        system_prompt = """
你是直播文稿清洗和错别字修正助手。
这是一位达人直播时的文稿，主要讲解婴幼儿奶粉。
请保留原始语义，只删除无意义口头填充、重复废话和明显无效语气词。
请修正错别字、ASR 同音误识别、奶粉品牌名和常见配方词错误。
不要新增原文没有的信息，不要扩写卖点，不要改写为广告文案。
输出不要包含 #、项目符号、空行或解释。
字幕文件内容不要按每句话换行，用自然连续段落输出。
""".strip()
        user_prompt = f"""
这是第 {chunk_index}/{total_chunks} 段直播文稿。
请输出清洗纠错后的文稿内容。

原文：
{text}
""".strip()
        result = self._call_chat_text(
            system_prompt,
            user_prompt,
            log_label="LLM transcript correction",
            temperature=0.1,
        )
        return self._collapse_transcript_text(result)

    def _summarize_cleaned_transcript(
        self,
        filename: str,
        cleaned_transcript: str,
        prompt: Optional[str],
    ) -> str:
        chunks = self._split_text_by_chars(
            cleaned_transcript,
            max(12000, min(MAX_LLM_INPUT_CHARS // 3, 50000)),
        )
        if not chunks:
            return "主题：婴幼儿奶粉选择\n总结：未生成可总结的文稿内容。"

        def summarize_chunk(job_index: int) -> str:
            index = job_index + 1
            chunk = chunks[job_index]
            system_prompt = """
你是直播带货内容总结助手。
只基于给定文稿总结，不得编造。
重点关注合生元派星/派星奶粉、婴幼儿奶粉选择、竞品对比、观众关于派星的提问和主播回答。
输出不要包含 # 或空行。
""".strip()
            user_prompt = f"""
文件名：{filename}
原始筛选要求：{prompt or "未提供"}
这是第 {index}/{len(chunks)} 段已清洗文稿。

请总结：
1. 本段是否出现派星卖点；如出现，列出 HMO、OPO、口感、冲泡、奶源、营养等具体信息。
2. 是否出现竞品对比；如出现，说明竞品特点，并突出派星相对优势。
3. 是否出现观众关于派星的问答；如出现，按“观众问：...；主播答：...”总结，保留具体问题和实际建议，例如“一段派星，二段喝什么？生长曲线好就继续喝派星，不用换”。
4. 如果没有派星卖点、竞品对比或派星问答，明确说明没有。

文稿：
{chunk}
""".strip()
            return self._call_chat_text(
                system_prompt,
                user_prompt,
                log_label="LLM transcript partial summary",
                temperature=0.2,
            )

        partial_summaries = self._run_text_jobs_in_order(
            len(chunks),
            summarize_chunk,
            log_label="LLM transcript partial summary",
        )

        system_prompt = """
你是直播带货文稿总结助手。
请把局部总结合并成最终总结，输出不要包含 # 或空行。
必须包含主题和总结。
总结中必须包含派星卖点、竞品对比、观众关于派星的问答；如果原文没有，则明确写“未提及”。
竞品对比要在忠于原文的前提下突出派星优势。
观众问答要保留真实问题和主播回答，优先总结喂养阶段、换段、换奶、适合宝宝情况、价格、断货、冲泡、便便、过敏或生长曲线等与派星相关的问题。
""".strip()
        user_prompt = f"""
文件名：{filename}
请生成 Word 文档中的“总结”部分，参考格式：
主题：婴幼儿奶粉选择
总结：
1. 派星卖点：...
2. 竞品对比：...
3. 观众问答：...

局部总结：
{chr(10).join(partial_summaries)}
""".strip()
        return self._call_chat_text(
            system_prompt,
            user_prompt,
            log_label="LLM transcript final summary",
            temperature=0.2,
        ).replace("#", "").strip()

    def generate_transcript_document(
        self,
        filename: str,
        segments: Sequence[Dict],
        prompt: Optional[str] = None,
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
        transcript = self._collapse_transcript_text(transcript)
        if not transcript:
            return {
                "filename": filename,
                "summary": "主题：婴幼儿奶粉选择\n总结：未生成可总结的文稿内容。",
                "transcript": "",
            }

        chunks = self._split_text_by_chars(
            transcript,
            max(12000, min(MAX_LLM_INPUT_CHARS // 3, 45000)),
        )
        corrected_chunks = self._run_text_jobs_in_order(
            len(chunks),
            lambda job_index: self._correct_transcript_chunk(
                chunks[job_index],
                job_index + 1,
                len(chunks),
            ),
            log_label="LLM transcript correction",
        )
        cleaned_transcript = self._collapse_transcript_text(" ".join(corrected_chunks))
        summary = self._summarize_cleaned_transcript(filename, cleaned_transcript, prompt)
        return {
            "filename": filename,
            "summary": summary,
            "transcript": cleaned_transcript,
        }

    def _normalize_relevance_level(self, value: Any) -> str:
        text = str(value or "").strip()
        if text in ALLOWED_RELEVANCE_LEVELS:
            return text
        if any(keyword in text for keyword in ("负", "差评", "批评", "吐槽", "不推荐")):
            return "负面"
        if any(keyword in text for keyword in ("强", "核心", "重点", "明确", "主讲")):
            return "强相关"
        if any(keyword in text for keyword in ("弱", "间接", "顺带", "带到", "轻度")):
            return "弱相关"
        return "仅提及"

    def _normalize_intent(self, value: Any) -> str:
        text = str(value or "").strip()
        if text in ALLOWED_INTENTS:
            return text
        if any(keyword in text for keyword in ("主动", "讲解", "介绍", "卖点", "讲产品")):
            return "主动讲解"
        if any(keyword in text for keyword in ("展示", "演示", "试用", "上手", "给你看")):
            return "正面展示"
        if "对比" in text:
            return "对比提及"
        if any(keyword in text for keyword in ("观众", "弹幕", "评论", "提问", "问答")):
            return "回答观众"
        return "闲聊提及"

    def _normalize_segment(self, seg: Dict) -> Dict:
        normalized = dict(seg)
        try:
            normalized["start"] = float(seg.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            normalized["start"] = 0.0
        try:
            normalized["end"] = float(
                seg.get("end", normalized["start"]) or normalized["start"]
            )
        except (TypeError, ValueError):
            normalized["end"] = normalized["start"]

        normalized["summary"] = str(seg.get("summary", "") or "").strip()
        raw_tags = seg.get("tags", [])
        if isinstance(raw_tags, str):
            tags_list = [raw_tags.strip()] if raw_tags.strip() else []
        elif isinstance(raw_tags, Sequence):
            tags_list = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        else:
            tags_list = []
        normalized["tags"] = tags_list[:3]
        normalized["relevance_level"] = self._normalize_relevance_level(
            seg.get("relevance_level")
        )
        normalized["intent"] = self._normalize_intent(seg.get("intent"))
        normalized["text"] = str(seg.get("text", "") or "").strip()
        return normalized

    def _has_valid_labels(self, seg: Dict) -> bool:
        return (
            seg.get("relevance_level") in ALLOWED_RELEVANCE_LEVELS
            and seg.get("intent") in ALLOWED_INTENTS
        )

    def _attach_segment_source_text(
        self, segments: List[Dict], subtitles: List[Dict]
    ) -> List[Dict]:
        if not segments:
            return segments

        enriched_segments: List[Dict] = []
        for seg in segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            text_parts = []
            for subtitle in subtitles:
                sub_start = float(subtitle.get("start", 0.0) or 0.0)
                sub_end = float(subtitle.get("end", sub_start) or sub_start)
                if sub_end < start or sub_start > end:
                    continue
                text = str(subtitle.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
            enriched = dict(seg)
            enriched["text"] = " ".join(text_parts).strip()
            enriched_segments.append(enriched)
        return enriched_segments

    def _filter_candidate_segments_by_validity(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return segments

        filtered_segments: List[Dict] = []
        removed = 0
        for seg in segments:
            if not isinstance(seg, dict):
                removed += 1
                continue

            try:
                start = float(seg.get("start", 0.0) or 0.0)
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(seg.get("end", start) or start)
            except (TypeError, ValueError):
                end = start

            summary = str(seg.get("summary", "") or "").strip()
            evidence = str(seg.get("evidence", "") or "").strip()
            if end < start or (not summary and not evidence):
                removed += 1
                continue

            normalized = dict(seg)
            normalized["start"] = start
            normalized["end"] = end
            normalized["summary"] = summary or evidence
            normalized["evidence"] = evidence
            normalized["text"] = str(seg.get("text", "") or "").strip()
            filtered_segments.append(normalized)

        logger.info(
            "LLM first-pass candidate filter segments %d -> %d (removed=%d)",
            len(segments),
            len(filtered_segments),
            removed,
        )
        return filtered_segments

    def _filter_segments_by_validity(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return segments

        filtered_segments: List[Dict] = []
        removed = 0
        for seg in segments:
            normalized = self._normalize_segment(seg)
            if normalized["end"] < normalized["start"]:
                removed += 1
                continue
            if not normalized["summary"] and not normalized["text"]:
                removed += 1
                continue
            if not self._has_valid_labels(normalized):
                removed += 1
                continue
            filtered_segments.append(normalized)

        logger.info(
            "LLM segment_video: validity filter segments %d -> %d (removed=%d)",
            len(segments),
            len(filtered_segments),
            removed,
        )
        return filtered_segments

    def _apply_filter_mode(
        self, segments: List[Dict], filter_mode: Optional[str] = None
    ) -> List[Dict]:
        if not segments:
            return segments

        allowed_levels = self._allowed_relevance_levels(filter_mode)
        filtered = [
            seg for seg in segments if seg.get("relevance_level") in allowed_levels
        ]
        logger.info(
            "LLM segment_video: filter mode %s segments %d -> %d",
            filter_mode or self.segment_filter_mode,
            len(segments),
            len(filtered),
        )
        return filtered

    def _build_user_prompt(self, user_prompt: Optional[str]) -> str:
        user_prompt = (user_prompt or "").strip()
        filter_mode_label = FILTER_MODE_DEFINITIONS[self.segment_filter_mode]["label"]
        guardrail = (
            "系统规则优先于用户补充说明。"
            f"当前结果保留范围固定为：{filter_mode_label}。"
            "如果用户说明里出现“只保留强相关”“忽略弱相关”之类与当前模式冲突的话，请不要提前删减候选，"
            "仍先完整召回所有派星相关片段，再由第二轮标签分类和程序按当前模式筛选。"
        )
        if not user_prompt:
            user_prompt = "请找出所有提及派星/合生元派星/合生元的片段，并标注准确标签。"
        return f"{guardrail}\n\n用户补充说明：{user_prompt}"

    def _run_second_pass_filter(
        self, segments: List[Dict], prompt: Optional[str] = None
    ) -> List[Dict]:
        if not segments:
            return segments

        confirmed_segments: List[Dict] = []
        task_hint = self._build_user_prompt(prompt)

        for batch_start in range(0, len(segments), SECOND_PASS_BATCH_SIZE):
            batch = segments[batch_start: batch_start + SECOND_PASS_BATCH_SIZE]
            payload = []
            for idx, seg in enumerate(batch):
                payload.append(
                    {
                        "index": idx,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", ""),
                        "summary": seg.get("summary", ""),
                        "evidence": seg.get("evidence", ""),
                        "anchor_terms": seg.get("anchor_terms", []),
                        "source_window_index": seg.get("source_window_index"),
                        "source_segment_indices": seg.get("source_segment_indices", []),
                        "source_global_segment_indices": seg.get(
                            "source_global_segment_indices", []
                        ),
                        "tags": seg.get("tags", []),
                        "relevance_level": seg.get("relevance_level", ""),
                        "intent": seg.get("intent", ""),
                    }
                )

            system_prompt = """
你是直播带货片段二轮筛选与标签分类员。候选片段来自“规则锚点召回 + 局部窗口内精细切段”，你现在只做判别和打标签。

规则：
1. 先判断候选片段是否确实提到或明确指向派星 / 合生元派星。单独出现“合生元”时，只有上下文明确是在讲派星产品线，才算相关；完全无关才 should_keep=false。
2. 不要只保留强相关。只要确实相关，强相关、弱相关、仅提及、负面都应该 should_keep=true，最终由程序按“结果保留范围”过滤。
3. 根据片段文本、摘要、证据、锚点词和 source_segment_indices，给每个保留片段打 final_relevance_level 与 final_intent。
4. final_relevance_level 只能是：强相关、弱相关、仅提及、负面。
5. final_intent 只能是：主动讲解、正面展示、对比提及、回答观众、闲聊提及。

分类参考：
- 强相关：片段主体明确在讲目标品牌/产品本身，比如配方、卖点、适用宝宝、价格、购买建议、换段建议等。
- 弱相关：片段与目标品牌有关，但主体不是完整讲派星，比如竞品对比里顺带提到、过渡话术、轻度关联。
- 仅提及：只是提到品牌名、产品名、链接号、有无货、价格一句话等，没有展开讲解。
- 负面：明确负面评价、质疑、吐槽、投诉、不推荐，或主播在讲派星缺点/风险/不适合。

输出要求：
- 只返回 JSON 数组。
- 每个元素只包含：index, should_keep, final_relevance_level, final_intent。
""".strip()

            user_prompt = (
                f"{task_hint}\n\n"
                "请复核下面这些候选片段：\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )

            reviewed = self._call_chat_json(
                system_prompt,
                user_prompt,
                log_label="LLM second-stage labeling",
                temperature=0.0,
            )
            if not isinstance(reviewed, list):
                raise ValueError("LLM second-stage labeling must return a JSON list")

            review_map = {}
            for item in reviewed:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if isinstance(idx, int):
                    review_map[idx] = item

            for idx, seg in enumerate(batch):
                reviewed_item = review_map.get(idx)
                if not reviewed_item:
                    continue
                should_keep_value = reviewed_item.get("should_keep")
                if isinstance(should_keep_value, str):
                    should_keep = should_keep_value.strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "是",
                        "保留",
                    )
                else:
                    should_keep = bool(should_keep_value)
                if not should_keep:
                    continue
                seg["relevance_level"] = self._normalize_relevance_level(
                    reviewed_item.get("final_relevance_level")
                )
                seg["intent"] = self._normalize_intent(
                    reviewed_item.get("final_intent")
                )
                confirmed_segments.append(seg)

        logger.info(
            "LLM second-stage labeling segments %d -> %d",
            len(segments),
            len(confirmed_segments),
        )
        return confirmed_segments

    def _finalize_candidate_segments(
        self,
        candidate_segments: List[Dict],
        prompt: Optional[str] = None,
        *,
        write_summary: bool = True,
    ) -> List[Dict]:
        reviewed_segments = self._run_second_pass_filter(candidate_segments, prompt)
        valid_segments = self._filter_segments_by_validity(reviewed_segments)

        filtered_segments = self._apply_filter_mode(
            valid_segments, self.segment_filter_mode
        )
        filtered_segments.sort(
            key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        )

        for seg in filtered_segments:
            seg.pop("text", None)

        if write_summary:
            with open(
                f"segment-data/summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(filtered_segments, f, ensure_ascii=False, indent=4)

        return filtered_segments

    def segment_anchor_window_candidates(
        self, window: Dict, prompt: Optional[str] = None
    ) -> List[Dict]:
        """Cut precise Paixing candidate clips inside one rule-recalled window."""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        window_index = window.get("window_index")
        window_start = float(window.get("start", 0.0) or 0.0)
        window_end = float(window.get("end", window_start) or window_start)
        anchor_terms = window.get("anchor_terms") or []
        source_segments = window.get("segments") or []

        payload_segments = []
        for idx, segment in enumerate(source_segments):
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

        if not payload_segments:
            return []

        task_hint = self._build_user_prompt(prompt)
        system_prompt = """
你是直播带货片段切分助手。输入已经由规则锚点召回限定在派星相关候选窗口内；你不需要在整场直播里搜索派星，只需要在当前窗口内部切出上下文完整、可以独立看懂的派星候选片段。

重要原则：
- 片段单位不是“最短命中句”，而是“观众能看懂主播在讲什么”的完整内容单元。
- 宁可比关键词句多保留一点必要上下文，也不要只保留“派星”出现的那一两句。
- 上下文包括：观众问题、主播承接、产品/品牌指代对象、卖点解释、对比对象、结论或建议。
- 遇到“它/这个/这款/这个奶粉/三段/一段”等指代词时，必须向前包含能说明指代对象的小句。
- 遇到问答时，必须同时包含问题和回答；只保留回答或只保留问题都不合格。
- 遇到卖点讲解或竞品对比时，必须保留能说明理由的解释链路，不能只切出结论句。

目标：
1. 从窗口内字幕小句中，切出所有确实提到或明确指向派星 / 合生元派星的片段。
2. 单独出现“合生元”时，只有上下文明确是在讲派星产品线，才作为派星相关候选。
3. 候选片段可以是强相关、弱相关、仅提及或负面，不要在本步骤提前删除；完全无关内容不要输出。
4. 不要把整个窗口直接作为结果。窗口只是上下文，最终片段必须是窗口内部的一段或多段完整内容单元。
5. start 应该落在“问题/话题引入/指代对象说明”开始处；end 应该落在“讲解结束/建议结束/话题切换”处。
6. 当同一派星话题连续讲解时，应合并为一个完整片段；不要按每个卖点、每句话拆成多个碎片。
7. 遇到明显换产品、换观众问题、纯闲聊、价格/库存无关插话时应断开。
8. start 必须等于输入 segments 中某条字幕的 start；end 必须等于输入 segments 中某条字幕的 end；禁止编造新时间。
9. source_segment_indices 使用输入 segments 的 idx，必须覆盖该候选片段实际使用的所有小句，通常应该是一段连续区间。

输出要求：
- 只返回 JSON 数组。
- 每个元素只包含：start, end, summary, evidence, source_segment_indices。
- summary 客观概括整个片段的上下文，不要只概括关键词句。
- evidence 写触发保留的原文证据、锚点说明，以及为什么这些上下文需要一起保留。
- 如果窗口内没有真实派星相关内容，返回 []。
""".strip()

        user_prompt = (
            f"{task_hint}\n\n"
            f"窗口信息：window_index={window_index}, window_start={window_start}, "
            f"window_end={window_end}, anchor_terms={json.dumps(anchor_terms, ensure_ascii=False)}\n"
            "下面是窗口内按时间排序的字幕小句 JSON，请按“上下文完整、可独立看懂”的标准切出派星候选片段：\n"
            f"{json.dumps(payload_segments, ensure_ascii=False)}"
        )

        candidate_segments = self._call_chat_json(
            system_prompt,
            user_prompt,
            log_label="LLM anchor-window fine segmentation",
        )
        if not isinstance(candidate_segments, list):
            raise ValueError("LLM anchor-window fine segmentation must return a JSON list")

        segment_by_idx = {
            int(segment["idx"]): segment
            for segment in payload_segments
            if isinstance(segment.get("idx"), int)
        }
        enriched_candidates = []
        for candidate in candidate_segments:
            if not isinstance(candidate, dict):
                continue
            candidate = dict(candidate)
            raw_indices = candidate.get("source_segment_indices") or []
            source_indices = []
            for item in raw_indices:
                try:
                    source_indices.append(int(item))
                except (TypeError, ValueError):
                    continue
            source_indices = sorted(
                idx for idx in set(source_indices) if idx in segment_by_idx
            )
            if source_indices:
                text = " ".join(
                    str(segment_by_idx[idx].get("text") or "").strip()
                    for idx in source_indices
                    if str(segment_by_idx[idx].get("text") or "").strip()
                )
            else:
                text = ""
                start = float(candidate.get("start", 0.0) or 0.0)
                end = float(candidate.get("end", start) or start)
                for segment in payload_segments:
                    sub_start = float(segment.get("start", 0.0) or 0.0)
                    sub_end = float(segment.get("end", sub_start) or sub_start)
                    if sub_end < start or sub_start > end:
                        continue
                    text_part = str(segment.get("text") or "").strip()
                    if text_part:
                        text = f"{text} {text_part}".strip()

            candidate["text"] = text
            candidate["source_segment_indices"] = source_indices
            candidate["source_global_segment_indices"] = [
                segment_by_idx[idx].get("segment_index", idx)
                for idx in source_indices
            ]
            candidate["source_window_index"] = window_index
            candidate["window_start"] = window_start
            candidate["window_end"] = window_end
            candidate["anchor_terms"] = anchor_terms
            enriched_candidates.append(candidate)

        valid_candidates = self._filter_candidate_segments_by_validity(
            enriched_candidates
        )
        valid_candidates.sort(
            key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        )
        return valid_candidates

    def recall_paixing_candidates(
        self, subtitles: Any, prompt: Optional[str] = None
    ) -> List[Dict]:
        """Run the first-stage Paixing candidate recall used by segment_video."""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        normalized_subtitles = self._normalize_subtitles(subtitles)
        subtitles_json = json.dumps(normalized_subtitles, ensure_ascii=False)
        filter_mode_label = FILTER_MODE_DEFINITIONS[self.segment_filter_mode]["label"]
        task_hint = self._build_user_prompt(prompt)

        system_prompt = f"""
你是直播带货片段召回助手。请从字幕中找出所有提及或明确指向目标品牌/产品的片段。

目标品牌包括：{", ".join(TARGET_BRAND_NAMES)}。

任务要求：
1. 第一轮只做“捞出候选片段”，不要判断好坏，不要打 relevance_level，不要打 intent。
2. 只要片段明确提到目标品牌/产品，或者结合上下文可以确认“这个/这款/它/链接号”等指向目标品牌/产品，就应输出。
3. 不管正面、负面、强相关、弱相关、仅提及，都要先召回。
4. 当前结果保留范围是：{filter_mode_label}。但本轮不要提前删减候选，最终筛选由第二轮标签分类和程序过滤完成。
5. start 和 end 必须来自原始字幕边界，不要编造新时间；需要覆盖完整语义，必要时可合并相邻字幕边界。
6. 如果片段完全没有品牌名，也无法从紧邻上下文确认指代目标品牌，则不要输出。

输出要求：
- 只返回 JSON 数组。
- 每个元素只包含：start, end, summary, evidence。
- summary 写客观概括，不写分析过程。
- evidence 摘录或概括触发召回的关键词/句子，例如“提到派星二段怎么喝”“这款指向前文派星”。
""".strip()

        full_prompt = (
            f"{task_hint}\n\n"
            "下面是按时间排序的字幕 JSON，请基于它返回结果：\n"
            f"{subtitles_json}"
        )

        logger.info(
            "LLM segment_video: normalized_segments=%d, prompt_chars=%d, raw_type=%s, two_stage_labeling=%s, legacy_second_pass_flag=%s, filter_mode=%s",
            len(normalized_subtitles),
            len(full_prompt) + len(system_prompt),
            type(subtitles).__name__,
            True,
            self.enable_second_pass_review,
            self.segment_filter_mode,
        )

        candidate_segments = self._call_chat_json(
            system_prompt,
            full_prompt,
            log_label="LLM first-pass candidate recall",
        )
        if not isinstance(candidate_segments, list):
            raise ValueError("LLM first-pass candidate recall must return a JSON list")

        enriched_candidates = self._attach_segment_source_text(
            candidate_segments, normalized_subtitles
        )
        valid_candidates = self._filter_candidate_segments_by_validity(
            enriched_candidates
        )
        valid_candidates.sort(
            key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        )
        return valid_candidates

    def segment_video(self, subtitles: Any, prompt: Optional[str] = None) -> List[Dict]:
        # Legacy full-subtitle first pass. Kept for rollback via
        # SEGMENT_RECALL_STRATEGY=legacy_chunk.
        valid_candidates = self.recall_paixing_candidates(subtitles, prompt)
        return self._finalize_candidate_segments(valid_candidates, prompt)

    def label_and_filter_candidates(
        self, candidates: List[Dict], prompt: Optional[str] = None
    ) -> List[Dict]:
        """Run the retained second-stage labeling and configured result filtering."""
        return self._finalize_candidate_segments(candidates, prompt)
