import os
import tempfile
import torch
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# 国内直连 Hugging Face 易 SSL 超时，默认用镜像以便 CTC 对齐等模型可下载；.env 中设 HF_ENDPOINT= 可改回官方
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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


def _torch_cuda_supports_device(device_index=0):
    """Return whether this PyTorch build has kernels for the CUDA device."""
    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
        current_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        return (
            current_arch in supported_arches
            or f"compute_{major}{minor}" in supported_arches
        )
    except Exception as exc:
        print(f"Unable to check PyTorch CUDA architecture support: {exc}")
        return True


def get_alignment_device_config():
    """Pick a safe device for PyTorch-based forced alignment."""
    override = os.getenv("ALIGNMENT_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda"):
        return override

    if DEVICE_TYPE != "cuda":
        return DEVICE_TYPE

    device_index = AVAILABLE_GPUS[0] if AVAILABLE_GPUS else 0
    if _torch_cuda_supports_device(device_index):
        return "cuda"

    try:
        gpu_name = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        supported = ", ".join(torch.cuda.get_arch_list()) or "unknown"
        print(
            "PyTorch CUDA does not support the alignment GPU "
            f"{gpu_name} (sm_{major}{minor}); supported arches: {supported}. "
            "Falling back to CPU for forced alignment. "
            "Set ALIGNMENT_DEVICE=cuda after installing a compatible PyTorch build."
        )
    except Exception:
        print(
            "PyTorch CUDA support for forced alignment could not be confirmed. "
            "Falling back to CPU for forced alignment."
        )
    return "cpu"


# 文件上传配置
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'ts', 'mxf', 'mp3', 'wav',
                      'flac']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_FILE_NUMBERS = 10  # 最大文件数量
# 单文件最长时长（秒），默认 6 小时；可通过环境变量 MAX_DURATION_SECONDS 覆盖
def _parse_max_duration():
    val = os.getenv("MAX_DURATION_SECONDS", "").strip()
    if not val:
        return 360 * 60  # 6 小时
    try:
        return int(val)
    except ValueError:
        return 360 * 60


MAX_DURATION_SECONDS = _parse_max_duration()

# 临时文件夹
TEMP_FOLDER = os.getenv(
    "PREENCUT_TEMP_DIR",
    os.path.join(tempfile.gettempdir(), "preencut-temp"),
)

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
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "32"))
FASTER_WHISPER_BEAM_SIZE = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "8"))
FASTER_WHISPER_USE_BATCHED_PIPELINE = (
    os.getenv("FASTER_WHISPER_USE_BATCHED_PIPELINE", "true").lower()
    not in ("0", "false", "no", "off")
)
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

# 短句合并：字数（字符数）低于此值的 segment 会与上一段合并，减少碎片；0 表示不合并。改小则合并更少、断句更细
MERGE_SEGMENT_MAX_CHARS = int(os.getenv("MERGE_SEGMENT_MAX_CHARS", "4"))

# 多文件时是否按 summary 跨文件去重（两段视频中部分内容重复时只保留一条片段）
DEDUPE_SEGMENTS_ACROSS_FILES = os.getenv("DEDUPE_SEGMENTS_ACROSS_FILES", "true").strip().lower() in ("1", "true", "yes")

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
        "对比后推荐其他",
        "对比后没选派星",
        "推荐了别的",
        "关注榜一",
        "小助理",
        "点关注不迷路",
        "点点赞",
        "扣个1",
        "送礼物",
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
ALIGNMENT_DEVICE = get_alignment_device_config()  # 对齐模型使用的设备
ALIGNMENT_MODEL = 'ctc-forced-aligner'  # 使用的对齐模型, whisperx, ctc-forced-aligner
ALIGNMENT_BATCH_SIZE = int(os.getenv("ALIGNMENT_BATCH_SIZE", "32"))
# 长音频对齐时按片段处理，单段最长秒数，超出则切分后分批对齐以免 MemoryError（默认 90 分钟）
_align_chunk_env = os.getenv("ALIGNMENT_CHUNK_DURATION_SECONDS", "").strip()
try:
    ALIGNMENT_CHUNK_DURATION_SECONDS = int(_align_chunk_env) if _align_chunk_env else 90 * 60
except ValueError:
    ALIGNMENT_CHUNK_DURATION_SECONDS = 90 * 60

# 大模型请求超时（秒），走代理或长视频时可调大，避免握手/等待超时
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "180"))
# 大模型瞬时故障重试：用于处理 429/503/5xx/超时等临时错误
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "5"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "60"))
# 单次 LLM 响应 token 上限兜底。Gemini 配置较大时，过高的 max_tokens 可能增加排队/高负载概率。
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "65536"))

ENABLE_SECOND_PASS_REVIEW = (
    os.getenv("ENABLE_SECOND_PASS_REVIEW", "false").strip().lower()
    in ("1", "true", "yes")
)

SEGMENT_FILTER_MODE = os.getenv(
    "SEGMENT_FILTER_MODE",
    "all_mentions",
).strip() or "all_mentions"

# 长视频调用控制：单次传入大模型的最大字幕条数与最大字符数
# 分片逻辑：先按 MAX_SEGMENTS_PER_LLM_CALL 切区间；若某片 JSON 字符数 > MAX_LLM_INPUT_CHARS，会二分缩小该片直至满足。
# chunk_chars 建议（按模型上下文）：32k 上下文约 15000～20000；128k 约 60000～80000；200k+ 可 100000～150000。
# 区间建议：单条 segment 的 JSON 约 50～60 字符，故 MAX_SEGMENTS_PER_LLM_CALL 与 MAX_LLM_INPUT_CHARS 需匹配，
# 例如 MAX_LLM_INPUT_CHARS=80000 时单片最多约 1300～1600 条，设为 1200 较稳妥；再大需同时提高 MAX_LLM_INPUT_CHARS。
MAX_SEGMENTS_PER_LLM_CALL = int(os.getenv("MAX_SEGMENTS_PER_LLM_CALL", "1000"))
MAX_LLM_INPUT_CHARS = int(os.getenv("MAX_LLM_INPUT_CHARS", "150000"))
LLM_TRANSCRIPT_CONCURRENCY = max(
    1,
    int(os.getenv("LLM_TRANSCRIPT_CONCURRENCY", "3")),
)
TRANSCRIPT_LLM_MODEL = os.getenv(
    "TRANSCRIPT_LLM_MODEL",
    "gemini-3.1-flash-lite",
).strip() or "gemini-3.1-flash-lite"
# 某个 LLM 分片在重试后仍失败时，若片段数大于该值，会自动拆成更小的片继续尝试。
LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS = int(os.getenv("LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS", "500"))
# 大模型分片重叠条数：相邻两片重叠若干条，避免边界处漏掉片段（如派星）；合并时按时间去重
_llm_overlap_env = os.getenv("LLM_CHUNK_OVERLAP_SEGMENTS", "").strip()
try:
    LLM_CHUNK_OVERLAP_SEGMENTS = int(_llm_overlap_env) if _llm_overlap_env else 200
except ValueError:
    LLM_CHUNK_OVERLAP_SEGMENTS = 200

# Segment recall strategy:
# - anchor_window: scan Paixing anchors, build local context windows, then ask LLM
#   to cut precise clips inside each window.
# - legacy_chunk: old full-subtitle chunking flow, kept as rollback path.
SEGMENT_RECALL_STRATEGY = os.getenv(
    "SEGMENT_RECALL_STRATEGY", "anchor_window"
).strip().lower() or "anchor_window"
ANCHOR_RECALL_PRE_SECONDS = float(os.getenv("ANCHOR_RECALL_PRE_SECONDS", "30"))
ANCHOR_RECALL_POST_SECONDS = float(os.getenv("ANCHOR_RECALL_POST_SECONDS", "90"))
ANCHOR_RECALL_MERGE_GAP_SECONDS = float(
    os.getenv("ANCHOR_RECALL_MERGE_GAP_SECONDS", "10")
)
ANCHOR_WINDOW_LLM_CONCURRENCY = max(
    1,
    int(os.getenv("ANCHOR_WINDOW_LLM_CONCURRENCY", "3")),
)
ANCHOR_RECALL_INCLUDE_HESHENGYUAN_ALONE = (
    os.getenv("ANCHOR_RECALL_INCLUDE_HESHENGYUAN_ALONE", "false").strip().lower()
    in ("1", "true", "yes")
)
ANCHOR_RECALL_TERMS = [
    "派星",
    "派新",
    "派心",
    "派芯",
    "派薪",
    "合生元派星",
    "合成元派星",
    "合成员派星",
    "合生源派星",
    "合生派星",
    "和声元派星",
    "和声园派星",
    "和尚元派星",
    "和尚园派星",
    "和尚派星",
    "核权派新",
    "派星一段",
    "派星二段",
    "派星三段",
]
_anchor_extra_terms = [
    term.strip()
    for term in os.getenv("ANCHOR_RECALL_EXTRA_TERMS", "").split(",")
    if term.strip()
]
if _anchor_extra_terms:
    ANCHOR_RECALL_TERMS.extend(_anchor_extra_terms)
if ANCHOR_RECALL_INCLUDE_HESHENGYUAN_ALONE:
    ANCHOR_RECALL_TERMS.append("合生元")
ANCHOR_RECALL_TERMS = sorted(set(ANCHOR_RECALL_TERMS), key=len, reverse=True)

# OpenAI API 配置（max_tokens 为单次响应上限，长视频/多片段时需足够大以免结果被截断）
LLM_MODEL_OPTIONS = [
    {
        "model": "gemini-3.1-pro-preview",
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
        "max_tokens": 16384
    },
    {
        "model": "doubao-1-5-pro-32k-250115",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env_name": "DOUBAO_1_5_PRO_API_KEY",
        "label": "豆包",
        "max_tokens": 16384
    }
]

TRANSCRIPT_LLM_MODEL_OPTIONS = [
    {
        "model": "gemini-3.1-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env_name": "GOOGLE_API_KEY",
        "label": "gemini-3.1-flash-lite",
        "max_tokens": 65536,
    }
]

# 创建必要的目录
for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
