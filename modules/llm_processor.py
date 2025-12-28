import json
import os
from datetime import datetime
from openai import OpenAI
from config import LLM_MODEL_OPTIONS
from typing import List, Dict, Optional


class LLMProcessor:
    def __init__(self, llm_model: str, temperature: float):
        for model in LLM_MODEL_OPTIONS:
            if model['label'] == llm_model:
                self.api_key = os.getenv(model['api_key_env_name'])
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=model['base_url'],
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

    def segment_video(self, subtitles: str,
                      prompt: Optional[str] = None) -> \
            List[Dict]:
        """使用大模型根据字幕内容进行视频分段"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        # 构建系统提示
        system_prompt = (
            f"""你是一个专业的视频剪辑助手，需要根据提供的字幕内容和用户要求将字幕处理成片段。
            注意：
            1. 字幕内容来自语音识别软件，可能存在大量的同音字或口语中口齿不清的文字，分析语义时请根据上下文或相近字（包括但不限于以下相似发音对照表）进行理解。
            2. 字幕内容来自网络视频平台的带货直播，字幕片段分为三类：产品介绍、观众和主播连线互动问答、主播与评论区留言互动问答。
            3. 处理字幕时，要考虑第2条的场景，将内容关联的字幕片段合并在一起，形成一个完整的片段。
            比如：同一个观众的连线互动问答，必须合并在一起，形成一个完整的片段（从打招呼开始到感谢再见后结束）；同一个产品的介绍，必须合并在一起，形成一个完整的片段；
            对于评论区留言互动，主播会先读出留言观众的ID名，后读出观众的留言内容最后解答，必须将这些内容合并在一起，形成一个完整的片段。

            相似发音对照表：
            派星 -> 派心
            派星 -> 派新
            派星 -> 派芯
            合生元 -> 合生源
            合生元 -> 核酸盐
            合生元 -> 合顺园
            合生元 -> 和顺安

            要求：
            1. 每个片段必须包含以下信息：开始时间(秒)，结束时间(秒)，一句话的内容摘要和1-3个主题标签。
            2. 单个片段最长尽量不要超过总长度的30%，但还是优先考虑主题连贯性。
            3. 返回格式必须是完整有效的JSON格式的片段数组，每个片段是一个字典列表，每个字典有四个键：start, end, summary, tags。"""
        )

        # 用户提示（如果有自定义提示则使用）
        user_prompt = prompt or (
            "请根据以下字幕内容，将视频分成不超过10个有意义的片段。"
            "每个片段应包含连贯的主题内容，并给出一句话摘要和1-3个主题标签。"
            "时间信息需要精确到秒。"
        )

        # 组合完整的提示
        full_prompt = f"{user_prompt}\n\n字幕内容：\n{subtitles}"

        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

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

            # 结果保存到segment-data/summary_{date_time}.json
            with open(f'segment-data/summary_{datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=4)

            return segments
        except json.JSONDecodeError:
            # 尝试直接解析为JSON
            try:
                segments = json.loads(result)
                return segments
            except:
                raise ValueError(
                    f"处理大模型结果出错, 大模型返回:{result}"
                )
