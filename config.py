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
MAX_DURATION_SECONDS = 90 * 60  # 最长视频时长90分钟 = 5400秒

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

# VAD 最小静音时长（毫秒），仅当设置时传入 transcribe，用于减少句中短暂静音导致的错误断句
# 例如 300–500；不设置则使用 faster-whisper 默认
VAD_MIN_SILENCE_DURATION_MS = os.getenv("VAD_MIN_SILENCE_DURATION_MS", "")
try:
    VAD_MIN_SILENCE_DURATION_MS = int(VAD_MIN_SILENCE_DURATION_MS) if VAD_MIN_SILENCE_DURATION_MS else None
except ValueError:
    VAD_MIN_SILENCE_DURATION_MS = None

# ASR 后同音字/专有名词纠错：键为错误写法，值为正确写法，对每条 segment 的 text 做整词替换
# 例如 {"派心": "派星", "益生君": "益生菌"}，留空或 None 表示不纠错
POST_ASR_CORRECTION_MAP = {
    "派心": "派星",
    "派新": "派星",
    "派芯": "派星",
    "派薪": "派星",
    "和声员": "合生元",
    "和声元": "合生元",
    "核实员": "合生元",
    "核权派新": "合生元派星",
    "核权": "合生元",
    "益生君": "益生菌",
    "益身菌": "益生菌",
    "一生菌": "益生菌",
}
# 若需通过环境变量关闭纠错，可在代码中根据环境变量覆盖为空 dict；此处保留默认纠错表

# 语音文字对齐模型
ENABLE_ALIGNMENT = True  # 是否启用对齐
ALIGNMENT_DEVICE = DEVICE_TYPE  # 对齐模型使用的设备
ALIGNMENT_MODEL = 'ctc-forced-aligner'  # 使用的对齐模型, whisperx, ctc-forced-aligner

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
