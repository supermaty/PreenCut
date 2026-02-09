import os
import torch
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# API 密钥从环境变量或 .env 文件读取
# 如果环境变量中不存在，则从 .env 文件中读取
# 注意：.env 文件中的值会覆盖已存在的环境变量（除非设置 override=False）


# 设置Gradio临时目录
os.environ['GRADIO_TEMP_DIR'] = '/data/tmp/gradio'


def get_available_gpus():
    """获取所有可用的GPU设备"""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def get_device_config():
    """设置设备配置"""
    gpus = get_available_gpus()

    # 检查环境变量是否指定了GPU
    cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        try:
            # 解析环境变量中的GPU索引
            selected_gpus = [int(x.strip()) for x in cuda_visible.split(',') if
                             x.strip()]
            return 'cuda', selected_gpus
        except ValueError:
            pass

    # 如果没有指定但检测到GPU
    if gpus:
        return 'cuda', gpus

    # 默认使用CPU
    return 'cpu', []


# 文件上传配置
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'ts', 'mxf', 'mp3', 'wav',
                      'flac']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_FILE_NUMBERS = 10  # 最大文件数量
MAX_DURATION_SECONDS = 180 * 60  # 最长视频时长180分钟 = 10800秒（3小时）

# 临时文件夹
TEMP_FOLDER = "temp"

# 输出文件夹
OUTPUT_FOLDER = "output"

# 语音识别模型配置
SPEECH_RECOGNIZER_TYPE = 'faster-whisper'  # whisperx, faster-whisper

DEVICE_TYPE, AVAILABLE_GPUS = get_device_config()
# Whisper配置
WHISPER_MODEL_SIZE = 'large-v3-turbo'  # 模型大小 (tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo)
WHISPER_DEVICE = DEVICE_TYPE
WHISPER_GPU_IDS = AVAILABLE_GPUS
WHISPER_COMPUTE_TYPE = 'float16' if WHISPER_DEVICE == 'cuda' else 'float32'  # float16, float32, int8
WHISPER_BATCH_SIZE = 16  # 批处理大小
FASTER_WHISPER_BEAM_SIZE = 10  # 越大识别越准但越慢，降低会略损准确率
WHISPER_LANGUAGE = "zh"  # 指定语言：zh=中文，None=自动检测，en=英文 等

# Whisper 识别时的上下文提示（可配置，利于减少同音字与专有名词错误）
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    "请使用简体中文输出。Add punctuation after end of each line. 就比如说，我要先去吃饭。Segment at end of each sentence. 内容多为母婴或保健品带货直播，常出现品牌与产品名、成分与功能描述。示例词与短语：合生元、合生元派星、派星、益生菌、配方、成分、营养、功能、DHA、叶黄素、乳铁蛋白、品牌露出、口播、产品介绍、画面展示、深度讲解、完整语境、商务核算。"
).strip() or None  # 空字符串时使用 None，Whisper 将不设 initial_prompt

# VAD 最小静音时长（毫秒），用于减少句中短暂静音导致的过度断句
# 默认 600ms（比 faster-whisper 默认更“耐心”一些，让同一段话尽量不要被拆太碎）
# 如需调整，可通过环境变量 VAD_MIN_SILENCE_DURATION_MS 覆盖
_vad_min_silence_env = os.getenv("VAD_MIN_SILENCE_DURATION_MS", "").strip()
try:
    if _vad_min_silence_env:
        VAD_MIN_SILENCE_DURATION_MS = int(_vad_min_silence_env)
    else:
        VAD_MIN_SILENCE_DURATION_MS = 600
except ValueError:
    VAD_MIN_SILENCE_DURATION_MS = 600

# ASR 后同音字/专有名词纠错：键为错误写法，值为正确写法，对每条 segment 的 text 做整词替换
# 例如 {"派心": "派星", "益生君": "益生菌"}，留空或 None 表示不纠错
POST_ASR_CORRECTION_MAP = {
    # 「派星」相关常见误识别
    "派心": "派星",
    "派新": "派星",
    "派芯": "派星",
    "派薪": "派星",
    "和珊派星": "合生元派星",
    "和尚派星": "合生元派星",
    "和尚元派星": "合生元派星",
    "和尚园派星": "合生元派星",
    "和声园派星": "合生元派星",
    "合生派星": "合生元派星",

    # 「合生元」本身相关
    "和声员": "合生元",
    "和声元": "合生元",
    "核实员": "合生元",
    "和尚元": "合生元",
    "和尚园": "合生元",
    "核权": "合生元",

    # 组合提及时的误识别
    "核权派新": "合生元派星",

    # 益生菌相关
    "益生君": "益生菌",
    "益身菌": "益生菌",
    "一生菌": "益生菌",
}
# 若需通过环境变量关闭纠错，可在代码中根据环境变量覆盖为空 dict；此处保留默认纠错表

# 短句合并：字数（字符数）低于此值的 segment 会与上一段合并，减少碎片；0 表示不合并
MERGE_SEGMENT_MAX_CHARS = int(os.getenv("MERGE_SEGMENT_MAX_CHARS", "8"))

# 大模型结果后过滤关键词（可选兜底层）
# 可通过环境变量 EXCLUDE_SUMMARY_KEYWORDS / EXCLUDE_TAGS_KEYWORDS 覆盖，使用逗号分隔字符串
_summary_exclude_env = os.getenv("EXCLUDE_SUMMARY_KEYWORDS", "")
if _summary_exclude_env:
    EXCLUDE_SUMMARY_KEYWORDS = [
        kw.strip() for kw in _summary_exclude_env.split(",") if kw.strip()
    ]
else:
    EXCLUDE_SUMMARY_KEYWORDS = [
        "差评",
        "吐槽",
        "投诉",
        "不好",
        "后悔",
        "翻车",
        "买不到",
        "不给卖",
        "缺货",
        "断货",
        "抢不到",
    ]

_tags_exclude_env = os.getenv("EXCLUDE_TAGS_KEYWORDS", "")
if _tags_exclude_env:
    EXCLUDE_TAGS_KEYWORDS = [
        kw.strip() for kw in _tags_exclude_env.split(",") if kw.strip()
    ]
else:
    EXCLUDE_TAGS_KEYWORDS = [
        "负面",
        "差评",
        "闲聊",
        "跑题",
        "缺货",
        "买不到",
        "不给卖",
    ]

# 语音文字对齐模型
ENABLE_ALIGNMENT = True  # 是否启用对齐
ALIGNMENT_DEVICE = DEVICE_TYPE  # 对齐模型使用的设备
ALIGNMENT_MODEL = 'ctc-forced-aligner'  # 使用的对齐模型, whisperx, ctc-forced-aligner

# 大模型请求超时（秒），走代理或长视频时可调大，避免握手/等待超时
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "180"))

# 长视频调用控制：单次传入大模型的最大字幕条数与最大字符数
# 已按最长 3h 视频适配：单次可传更多内容，减少分片次数与连接中断概率；过大仍可能超时
MAX_SEGMENTS_PER_LLM_CALL = int(os.getenv("MAX_SEGMENTS_PER_LLM_CALL", "1200"))
MAX_LLM_INPUT_CHARS = int(os.getenv("MAX_LLM_INPUT_CHARS", "300000"))

# OpenAI API配置
LLM_MODEL_OPTIONS = [
    {
        "model": "gemini-3-pro-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env_name": "GOOGLE_API_KEY",
        "label": "gemini-3",
        "max_tokens": 200000
    },
    {
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env_name": "DEEPSEEK_V3_API_KEY",
        "label": "deepseek-reasoner-v3.2",
        "max_tokens": 4096
    },
    {
        "model": "doubao-1-5-pro-32k-250115",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env_name": "DOUBAO_1_5_PRO_API_KEY",
        "label": "豆包",
        "max_tokens": 4096
    }
]

# 创建必要的目录
for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
