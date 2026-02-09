import json
import logging
import os
import time
from datetime import datetime
from openai import OpenAI, APIConnectionError
from config import (
    LLM_MODEL_OPTIONS,
    LLM_REQUEST_TIMEOUT,
    EXCLUDE_SUMMARY_KEYWORDS,
    EXCLUDE_TAGS_KEYWORDS,
)
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMProcessor:
    def __init__(self, llm_model: str, temperature: float):
        for model in LLM_MODEL_OPTIONS:
            if model['label'] == llm_model:
                self.api_key = os.getenv(model['api_key_env_name'])
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=model['base_url'],
                    timeout=float(LLM_REQUEST_TIMEOUT),
                )
                self.model = model['model']
                self.temperature = temperature
                self.max_tokens = model.get('max_tokens', 4096)
                break

        if not hasattr(self, 'client'):
            raise ValueError(
                f"Unsupported LLM model: {llm_model}. Available models: "
                f"{', '.join([m['label'] for m in LLM_MODEL_OPTIONS])}"
            )

    def _normalize_subtitles(self, subtitles: Any) -> List[Dict]:
        """
        将各种可能的 subtitles 形式统一规范为
        List[Dict[start: float, end: float, text: str]] 结构，便于统一传给大模型。
        """
        try:
            # 字符串：尝试按 JSON 解析，否则作为单条文本兜底
            if isinstance(subtitles, str):
                s = subtitles.strip()
                if not s:
                    return []
                try:
                    loaded = json.loads(s)
                    return self._normalize_subtitles(loaded)
                except Exception:
                    return [{
                        "start": 0.0,
                        "end": 0.0,
                        "text": s,
                    }]

            # list：通常是若干 dict，或若干纯文本
            if isinstance(subtitles, list):
                normalized: List[Dict] = []
                for item in subtitles:
                    if isinstance(item, dict):
                        # 来自 llm_inputs 或对齐结果的 segment
                        start = item.get("start", 0.0) or 0.0
                        end = item.get("end", start)
                        text = str(item.get("text", "") or "")
                        if text:
                            try:
                                start_f = float(start)
                            except (TypeError, ValueError):
                                start_f = 0.0
                            try:
                                end_f = float(end)
                            except (TypeError, ValueError):
                                end_f = start_f
                            normalized.append({
                                "start": start_f,
                                "end": end_f,
                                "text": text,
                            })
                    else:
                        # 纯文本列表，按单条处理
                        text = str(item)
                        if text:
                            normalized.append({
                                "start": 0.0,
                                "end": 0.0,
                                "text": text,
                            })
                return normalized

            # dict：如 align_result，优先从 segments 中抽取
            if isinstance(subtitles, dict):
                segments = subtitles.get("segments")
                if isinstance(segments, list):
                    return self._normalize_subtitles(segments)

                # 没有 segments 时，将可见文本字段合并成一条
                text_parts = []
                for value in subtitles.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                if text_parts:
                    text = "\n".join(text_parts)
                else:
                    # 兜底：直接序列化整个 dict
                    text = json.dumps(subtitles, ensure_ascii=False)
                return [{
                    "start": 0.0,
                    "end": 0.0,
                    "text": text,
                }]
        except Exception as e:
            logger.exception("规范化字幕输入失败，将退化为单条文本: %s", e)

        # 最兜底：直接把对象转成字符串
        return [{
            "start": 0.0,
            "end": 0.0,
            "text": str(subtitles),
        }]

    def _filter_segments_by_keywords(self, segments: List[Dict]) -> List[Dict]:
        """
        在大模型结果之上做一层关键词过滤：
        - 若 summary 中命中 EXCLUDE_SUMMARY_KEYWORDS，则丢弃该片段；
        - 若 tags 中命中 EXCLUDE_TAGS_KEYWORDS，则丢弃该片段。
        同时将 tags 统一规范为字符串列表。
        """
        if not segments:
            return segments

        filtered_segments: List[Dict] = []
        removed = 0

        for seg in segments:
            summary = str(seg.get("summary", "") or "")

            raw_tags = seg.get("tags", [])
            if isinstance(raw_tags, str):
                tags_list = [raw_tags]
            elif isinstance(raw_tags, list):
                tags_list = [str(t) for t in raw_tags]
            elif raw_tags is None:
                tags_list = []
            else:
                tags_list = [str(raw_tags)]

            # 关键词命中则丢弃
            if EXCLUDE_SUMMARY_KEYWORDS:
                if any(kw in summary for kw in EXCLUDE_SUMMARY_KEYWORDS):
                    removed += 1
                    continue

            joined_tags = " ".join(tags_list)
            if EXCLUDE_TAGS_KEYWORDS:
                if any(kw in joined_tags for kw in EXCLUDE_TAGS_KEYWORDS):
                    removed += 1
                    continue

            seg["summary"] = summary
            seg["tags"] = tags_list
            filtered_segments.append(seg)

        logger.info(
            "LLM segment_video: keyword filter segments %d -> %d (removed=%d)",
            len(segments),
            len(filtered_segments),
            removed,
        )
        return filtered_segments

    def segment_video(self, subtitles: Any,
                      prompt: Optional[str] = None) -> List[Dict]:
        """使用大模型根据字幕内容进行视频分段"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        # 先将字幕统一规范成 JSON 友好的结构
        normalized_subtitles = self._normalize_subtitles(subtitles)
        subtitles_json = json.dumps(normalized_subtitles, ensure_ascii=False)

        # 构建系统提示
        system_prompt = (
            f"""你是一个专业的视频剪辑助手，需要根据提供的字幕内容和用户要求将字幕处理成片段。
            注意：
            1. 字幕内容来自语音识别软件，可能存在大量的同音字或口语中口齿不清的文字，分析语义时请根据上下文或相近字（包括但不限于以下相似发音对照表）进行理解。
            2. 字幕内容来自网络视频平台的带货直播，字幕片段分为三类：产品介绍、观众和主播连线互动问答、主播与评论区留言互动问答。
            3. 处理字幕时，要考虑第2条的场景，将内容关联的字幕片段合并在一起，形成一个完整的片段。
            比如：
            （1）同一个观众的连线互动问答，必须合并在一起，形成一个完整的片段（从打招呼开始到感谢再见后结束）；
            （2）同一个产品的介绍，必须合并在一起，形成一个完整的片段；
            （3）对于评论区留言互动，主播会先读出留言观众的ID名，后读出观众的留言内容最后解答，必须将这些内容合并在一起，形成一个完整的片段。

            相似发音对照表（分析字幕时按正确写法理解）：
            品牌与产品：
            派星 -> 派心、派新、派芯、派薪；
            合生元 -> 合生源、核酸盐、合顺园、和顺安、合生园；
            合生元派星 -> 合生元派心、合生源派星、合生元派薪。
            成分与营养：
            益生菌 -> 益生君、益身菌、一生菌；
            DHA -> D H A、底下去、DHA藻油；
            叶黄素 -> 叶黄素酯；
            乳铁蛋白 -> 乳铁、乳贴蛋白、乳铁旦白；
            核苷酸 -> 核甘酸、核干酸；
            OPO -> O P O、欧破欧。
            场景与动作：
            口播 -> 口波、口伯；
            品牌露出 -> 品牌漏出、品牌露初；
            直播间 -> 直波间；
            宝妈 -> 宝吗、保妈；
            奶粉 -> 乃粉、奶分。

            品牌露出与口播规则（优先执行）：
            - 目标：找出所有关于「合生元」及「合生元派星」的品牌露出和口播片段。
            - 必须包含：关键词提及的前后完整语境、产品功能深度讲解、成分描述以及画面展示部分。
            - 特别指令：对于长段落的产品介绍，必须提取完整的中间讲述过程，严禁只截取开头结尾。
            - 执行策略为「宁多勿少」：凡是涉及该品牌或产品的上下文关联内容（包括铺垫和总结），请全部保留，确保内容完整性以供商务核算。

            断句与边界规则（必须遵守）：
            - 输入给你的字幕是一个 JSON 数组 subtitles，其中每个元素都包含：start（秒）、end（秒）、text（识别文本）。
            - 你在生成结果片段时，片段的开始时间 start 必须严格等于某条字幕的 start，结束时间 end 必须严格等于某条字幕的 end，禁止在一条字幕的中间随意切时间。
            - 若某段内容跨多句，片段必须由若干完整字幕组成：start 取该段第一条字幕的 start，end 取该段最后一条字幕的 end，保证整句完整，不出现半句被截断的情况。
            - 结果片段必须按时间顺序排序，片段之间不能时间重叠，可以中间有空白间隔，但严禁交叉覆盖。
            - 严禁“缺头少尾”：不能只截取中间几句而故意去掉同一段话的开头或结尾。

            排除与筛选规则（分两步执行）：
            第一步：从字幕 JSON 中先排除以下内容：
            - 所有与「合生元」「合生元派星」无关的闲聊、寒暄、跑题聊天、无关紧要的废话。
            - 所有对「合生元」「合生元派星」的负面评价、投诉、吐槽、差评类言论，
              以及明显带有“买不到、不给卖、缺货、断货、抢不到”等负面情绪的句子或段落。
              例如（仅示例）：
              - “这个牌子真的不好用”“买了很后悔”“感觉完全没效果”“派星不给我卖”“派星老是缺货买不到”等，都不要出现在结果中。
            第二步：仅在剩余的正面或中性内容中进行片段合并：
            - 根据产品介绍、功能讲解、成分说明、画面展示、品牌露出等，将同一主题的连续字幕合并成完整片段。
            - 仍需遵守上面的断句与边界规则，以及“宁多勿少”的原则，为后续商务核算保留足够上下文。

            要求：
            1. 每个片段必须包含以下信息：开始时间(秒)，结束时间(秒)，一句话的内容摘要和1-3个主题标签。
            2. 单个片段最长尽量不要超过总长度的30%，但还是优先考虑主题连贯性。
            3. 你收到的 subtitles 可能只是整场直播的一部分，你只需要在这一部分内进行分段，不要去猜测未给出的前后内容。
            4. 返回格式必须是完整有效的 JSON 片段数组，每个片段是一个字典，且只能包含以下键：start, end, summary, tags，其中：
               - start: 浮点数或整数，表示片段开始时间（秒，必须来自输入 subtitles 的某条 start）。
               - end: 浮点数或整数，表示片段结束时间（秒，必须来自输入 subtitles 的某条 end）。
               - summary: 字符串，对该片段内容的一句话概括。
               - tags: 字符串数组（例如 ['品牌露出','产品讲解']），不要使用其它类型或嵌套结构。
            5. 严禁在 JSON 结果外输出多余解释性文字，如需解释请写入 summary 或 tags 中。"""
        )

        # 用户提示（如果有自定义提示则使用）
        user_prompt = prompt or (
            "请根据以下字幕内容，将视频分成不超过10个有意义的片段。"
            "每个片段应包含连贯的主题内容，并给出一句话摘要和1-3个主题标签。"
            "时间信息需要精确到秒。"
        )

        # 组合完整的提示，明确说明字幕 JSON 结构
        full_prompt = (
            f"{user_prompt}\n\n"
            "下面是本次需要分析的字幕 JSON 数组，变量名为 subtitles。\n"
            "subtitles 中的每个元素都包含 start（秒）、end（秒）、text（识别文本）。\n"
            "请严格使用这些时间戳作为片段的开始和结束边界，不要自己编造新的时间戳。\n"
            f"{subtitles_json}"
        )

        # 请求规模日志（便于排查超长/连接断开）
        logger.info(
            "LLM segment_video: normalized_segments=%d, prompt_chars=%d, raw_type=%s",
            len(normalized_subtitles),
            len(full_prompt),
            type(subtitles).__name__,
        )

        # 调用OpenAI API
        # 调用OpenAI API（带有限次数重试，主要针对网络/连接类错误）
        max_retries = 3
        last_exception: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                break
            except APIConnectionError as e:
                last_exception = e
                logger.warning(
                    "LLM API 连接错误(APIConnectionError)，第 %d/%d 次重试，prompt_chars=%d，错误类型=%s",
                    attempt,
                    max_retries,
                    len(full_prompt),
                    type(e).__name__,
                )
                if attempt < max_retries:
                    # 简单线性退避
                    time.sleep(2 * attempt)
                    continue
                logger.exception(
                    "LLM API 在重试后仍然连接失败(APIConnectionError)，prompt_chars=%d",
                    len(full_prompt),
                )
                raise
            except Exception as e:
                # 其它类型错误不做重试，直接抛出
                logger.exception(
                    "LLM API 调用失败(非连接错误): %s (prompt_chars=%d)",
                    type(e).__name__,
                    len(full_prompt),
                )
                raise

        # 解析响应
        result = response.choices[0].message.content

        # result = ''
        # # 读取segment-data/summary.json文件
        # with open('segment-data/summary.json', 'r', encoding='utf-8') as f:
        #     result = json.load(f)


        # 尝试提取JSON内容
        try:
            if result is None:
                raise ValueError("视频过长，请分段处理后再重新分析。")
            # 去除可能的代码块标记
            elif result.startswith("```json"):
                result = result[7:-3].strip()
            elif result.startswith("```"):
                result = result[3:-3].strip()

            segments = json.loads(result)
            segments = self._filter_segments_by_keywords(segments)

            # 结果保存到segment-data/summary_{date_time}.json
            with open(f'segment-data/summary_{datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=4)

            return segments
        except json.JSONDecodeError:
            # 尝试直接解析为JSON
            try:
                segments = json.loads(result)
                segments = self._filter_segments_by_keywords(segments)
                return segments
            except:
                raise ValueError(
                    f"处理大模型结果出错, 大模型返回:{result}"
                )
