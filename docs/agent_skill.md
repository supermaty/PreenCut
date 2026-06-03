---
name: preencut-livestream-clipping
description: >
  直播带货视频处理：上传视频/音频后，执行音频抽取、ASR、CTC 文本对齐、文稿级 LLM 纠错总结、
  品牌相关片段识别与标签分类，并导出字幕、CSV、总结文稿和 transcript_report.docx。
  Use when the user asks to process livestream videos, find brand/product mentions,
  clip or audit 派星/合生元派星相关内容, generate subtitles, generate brand-related
  summaries, or return timestamped candidate segments from a livestream recording.
metadata:
  author: "GGN"
  version: "1.0.0"
  repo: "local:D:/source-code/open-source/PreenCut"
  openapi: "docs/agent_skill_openapi.yaml"
---

# PreenCut Livestream Clipping

## 1. 系统概述

PreenCut 是一个面向直播带货视频的处理服务。它可以把长视频转成带时间戳的字幕，再用 LLM 找出目标品牌/产品相关片段，并给每个片段打上 `relevance_level` 与 `intent` 标签。

| 组件 | 技术/文件 | 作用 |
| --- | --- | --- |
| 服务入口 | `main.py` | 启动 FastAPI + Gradio，默认端口 `7861` |
| Agent API | `web/api.py` | `/api/agent/*`，供外部 agent 上传、轮询、取结果、下载产物 |
| Web UI | `web/gradio_ui.py` | 人工上传、查看任务、筛选片段、手动剪辑 |
| 队列 | `modules/processing_queue.py` | 串行执行视频处理任务并维护进度 |
| 音频/剪辑 | `modules/video_processor.py` + FFmpeg | 提取 WAV、按时间戳剪辑视频 |
| ASR | `modules/speech_recognizers/` | 默认 `faster-whisper`，模型 `large-v3-turbo` |
| 对齐 | `modules/aligners/text_aligner.py` | 默认 `ctc-forced-aligner`，细化字幕时间戳 |
| LLM 分析 | `modules/llm_processor.py` | 文稿级纠错总结、首轮候选召回、二轮标签筛选 |
| 输出 | `output/{task_id}/agent/` | `result.csv`、`summary_report.txt`、`transcript_report.docx`、字幕文件 |

## 2. 适用场景

使用这个 skill 当用户提出以下需求：

- “处理/分析这场直播视频”
- “找出所有提及派星/合生元派星的片段”
- “只要强相关、正面展示的可剪内容”
- “导出字幕、总结文稿、时间戳清单”
- “根据直播视频返回可以剪辑的片段”
- “审计一场直播里某品牌被怎么讲、被问了什么、是否有负面”

不要把它当成通用视频理解工具。当前链路主要依赖音频文本与 LLM，视觉内容不会被自动识别。

## 3. 前置条件

| 项目 | 要求 |
| --- | --- |
| 项目路径 | `D:\source-code\open-source\PreenCut` |
| Python 环境 | 推荐 `conda activate preencut` |
| FFmpeg | `ffmpeg` 与 `ffprobe` 必须在 PATH 中 |
| LLM Key | `.env` 中至少配置所选模型对应 API Key，例如 `GOOGLE_API_KEY` |
| 输入格式 | `mp4, avi, mov, mkv, ts, mxf, mp3, wav, flac` |
| 文件大小 | 默认最大 10GB |
| 默认服务地址 | `http://127.0.0.1:7861` |

外部平台 agent 如果不和 PreenCut 在同一台机器上，必须用 multipart 上传文件；只有当文件路径位于 PreenCut 服务所在机器上时，才能使用 `from-path` 接口。

## 4. 启动与健康检查

### 启动服务

```powershell
cd D:\source-code\open-source\PreenCut
conda activate preencut
python main.py
```

默认输出中应看到：

```text
Uvicorn running on http://0.0.0.0:7861
```

如果 `7861` 被占用，可以用 uvicorn 指定临时端口：

```powershell
cd D:\source-code\open-source\PreenCut
conda activate preencut
uvicorn main:app --host 0.0.0.0 --port 7862
```

### 健康检查

```powershell
curl.exe -s http://127.0.0.1:7861/api/agent/health
```

成功响应至少应包含：

```json
{
  "status": "ok",
  "service": "PreenCut Agent API",
  "default_llm_model": "gemini-3",
  "default_whisper_model_size": "large-v3-turbo",
  "filter_modes": ["all_mentions", "strong_only", "strong_and_weak", "non_negative_mentions"]
}
```

## 5. Agent 标准流程

```text
用户上传/提供视频
  -> Agent 检查 PreenCut 健康状态
  -> 创建任务：multipart 上传或 from-path
  -> 轮询 /api/agent/tasks/{task_id}
  -> status=completed 后请求 /result
  -> 下载 artifacts
  -> 根据用户需求返回总结、片段清单、附件，必要时用 ffmpeg 生成视频切片
```

内部处理链路：

```text
视频/音频输入
  -> FFmpeg 提取 16kHz 单声道 WAV
  -> faster-whisper ASR，得到 raw_text 与粗时间戳
  -> 可选 CTC forced alignment，细化字幕时间戳
  -> 短句合并
  -> 保留 raw_text，并生成 corrected_text 字段
     当前 corrected_text 只做固定词表替换，不再加载本地 AI 纠错模型
  -> LLM 生成整篇 transcript_report.docx：总结 + 文稿级纠错文本
  -> LLM 第一轮基于带时间戳字幕做品牌候选片段召回，只捞派星相关候选，不打标签
  -> LLM 第二轮对候选片段做标签分类：强相关、弱相关、仅提及、负面，并判断 intent
  -> 按结果保留范围过滤
  -> 生成 summary_report、CSV、TXT、SRT、DOCX 等产物
```

注意：`transcript_report.docx` 是整篇文稿级纠错和总结产物；为了避免时间戳漂移，分段和字幕时间戳仍基于原始带时间戳字幕字段，而不是把整篇纠错文稿重新切回时间线。

## 6. 创建任务

### 方式 A：文件已在 PreenCut 服务器上

优先用于本机 agent 或远程 agent 已把文件放到 PreenCut 机器上的情况。

```powershell
$body = @{
  file_path = "D:\data\video.mp4"
  llm_model = "gemini-3"
  prompt = "目标品牌：派星、合生元派星、合生元。请找出所有明确提及目标品牌/产品的直播片段，并标注相关度和意图。"
  whisper_model_size = "large-v3-turbo"
  temperature = 1.0
  enable_alignment = $true
  max_line_length = 32
  segment_filter_mode = "all_mentions"
  enable_second_pass_review = $false  # 兼容旧参数；二轮标签筛选现在固定执行
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:7861/api/agent/tasks/from-path" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

响应：

```json
{
  "task_id": "task_xxx",
  "status": "queued",
  "message": "local file accepted and queued",
  "poll_url": "http://127.0.0.1:7861/api/agent/tasks/task_xxx",
  "result_url": "http://127.0.0.1:7861/api/agent/tasks/task_xxx/result"
}
```

### 方式 B：multipart 上传文件

用于飞书、企业微信、浏览器插件、远程 agent 等无法直接访问服务器本地路径的场景。

```powershell
curl.exe -X POST "http://127.0.0.1:7861/api/agent/tasks" `
  -F "file=@D:\data\video.mp4" `
  -F "llm_model=gemini-3" `
  -F "prompt=目标品牌：派星、合生元派星、合生元。请输出所有提及目标品牌的片段。" `
  -F "whisper_model_size=large-v3-turbo" `
  -F "temperature=1.0" `
  -F "enable_alignment=true" `
  -F "max_line_length=32" `
  -F "segment_filter_mode=all_mentions" `
  -F "enable_second_pass_review=false"
```

## 7. 轮询与取结果

### 轮询状态

```powershell
$taskId = "task_xxx"
do {
  $status = Invoke-RestMethod "http://127.0.0.1:7861/api/agent/tasks/$taskId"
  $status | ConvertTo-Json -Depth 5
  Start-Sleep -Seconds 10
} while ($status.status -in @("queued", "processing"))

if ($status.status -eq "error") {
  throw $status.error
}
```

常见状态：

| status | 含义 |
| --- | --- |
| `queued` | 已排队 |
| `processing` | 正在 ASR、对齐、LLM 分析或生成产物 |
| `completed` | 完成，可取结果 |
| `error` | 失败，查看 `error` 字段 |
| `cancelled` | 用户取消 |
| `not_found` | 任务不存在或服务重启后内存状态丢失 |

### 获取结构化结果

```powershell
$result = Invoke-RestMethod "http://127.0.0.1:7861/api/agent/tasks/$taskId/result"
$result.segments | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 ".\segments.json"
```

`segments` 中每条片段包含：

| 字段 | 含义 |
| --- | --- |
| `filename` | 来源文件名 |
| `start` / `end` | 秒级时间戳 |
| `start_hms` / `end_hms` | `HH:MM:SS` 格式时间 |
| `duration` | 秒 |
| `summary` | 片段摘要 |
| `tags` | 1-3 个简短标签 |
| `relevance_level` | `强相关`、`弱相关`、`仅提及`、`负面` |
| `intent` | `主动讲解`、`正面展示`、`对比提及`、`回答观众`、`闲聊提及` |

## 8. 下载产物

```powershell
New-Item -ItemType Directory -Force ".\preencut-output" | Out-Null

foreach ($artifact in $result.artifacts) {
  $out = Join-Path ".\preencut-output" $artifact.name
  Invoke-WebRequest -Uri $artifact.download_url -OutFile $out
}
```

常见 artifact：

| 文件 | 用途 |
| --- | --- |
| `result.csv` | 保留片段清单，含时间、摘要、标签、相关度、意图；使用 UTF-8 BOM，Excel 可直接打开 |
| `summary_report.txt` | 所有保留片段的品牌相关内容总结 |
| `transcript_report.docx` | 整篇直播文稿级输出，包含总结和纠错后的文稿内容 |
| `{原文件名}.txt` | ASR 文本 |
| `{原文件名}.srt` | 字幕文件，开启对齐时基于对齐结果 |
| `{原文件名}_corrected.txt/.srt` | 仅当 `corrected_text` 与 `raw_text` 有差异时生成 |

## 9. 结果保留范围

`segment_filter_mode` 控制第二轮标签分类后最终保留哪些片段：

| 模式 | 中文含义 | 保留的 relevance_level | 推荐场景 |
| --- | --- | --- | --- |
| `all_mentions` | 全部相关提及（含负面） | 强相关、弱相关、仅提及、负面 | 品牌审计、完整复盘、找出所有派星出现位置 |
| `strong_only` | 仅强相关 | 强相关 | 只要最适合剪辑的核心讲解片段 |
| `strong_and_weak` | 强相关 + 弱相关 | 强相关、弱相关 | 想保留品牌讲解和轻度关联内容 |
| `non_negative_mentions` | 强相关 + 弱相关 + 仅提及 | 强相关、弱相关、仅提及 | 需要排除负面，但保留所有非负面提及 |

默认路线是 `all_mentions`，即输出所有提及派星/合生元派星/合生元的片段，并保留负面片段用于审计。

如果用户要求“只要正面可剪内容”，建议：

```json
{
  "segment_filter_mode": "strong_only",
  "enable_second_pass_review": false
}
```

取回结果后，agent 再做一次本地条件过滤：

```text
relevance_level == "强相关"
AND intent in ["主动讲解", "正面展示"]
```

当前 API 的 `segment_filter_mode` 只过滤 `relevance_level`，不直接过滤 `intent`。

## 10. 二轮筛选策略

第二轮标签分类现在是固定流程，不再作为前端独立开关使用。`enable_second_pass_review` 仅保留为兼容旧 API 参数；实际分段阶段总是执行：

```text
第一轮：宽松召回所有派星/合生元派星/合生元候选片段
第二轮：对候选片段判定是否确实相关，并打 relevance_level / intent 标签
程序过滤：按“结果保留范围”保留强相关、弱相关、仅提及、负面中的指定集合
```

这样做的目的，是让第一轮尽量不漏掉片段，让第二轮专注做标签分类，避免一个 prompt 同时承担“找片段”和“精细分类”导致漏召回或标签漂移。

## 11. 长文本处理

如果 ASR 字幕太长，系统不会一次性塞进单个 LLM 上下文。它会按以下配置自动切块：

| 配置 | 默认 | 含义 |
| --- | --- | --- |
| `MAX_SEGMENTS_PER_LLM_CALL` | `1000` | 每次 LLM 分段最多字幕段数，降低长视频单次请求压力 |
| `MAX_LLM_INPUT_CHARS` | `150000` | 每次 LLM 输入最大字符数 |
| `LLM_CHUNK_OVERLAP_SEGMENTS` | `200` | 分片之间的重叠字幕段数，降低边界漏召回 |
| `LLM_MAX_RETRIES` | `5` | 429/503/5xx/超时等临时错误的自动重试次数 |
| `LLM_FAILED_CHUNK_SPLIT_MIN_SEGMENTS` | `500` | 某个分片重试后仍失败时，大于该值会继续拆小重试 |

分片结果会按时间去重后合并。如果上游模型返回 `503 high demand`，系统会先指数退避重试；重试后仍失败的大分片会自动拆成更小的片继续尝试。

## 12. 可选：生成实际视频切片

Agent API 当前默认返回结构化片段和文档产物，不自动下载切好的 MP4。若用户明确要“视频切片文件”，agent 可以基于 `segments` 和原始视频路径调用 FFmpeg。

PowerShell 示例：

```powershell
$inputVideo = "D:\data\video.mp4"
$outDir = ".\preencut-clips"
New-Item -ItemType Directory -Force $outDir | Out-Null

$segmentsToClip = $result.segments | Where-Object {
  $_.relevance_level -eq "强相关" -and
  @("主动讲解", "正面展示") -contains $_.intent
}

$i = 1
foreach ($seg in $segmentsToClip) {
  $out = Join-Path $outDir ("clip_{0:000}.mp4" -f $i)
  ffmpeg -y `
    -ss $seg.start `
    -i $inputVideo `
    -t $seg.duration `
    -c:v libx264 `
    -c:a copy `
    -avoid_negative_ts make_zero `
    $out
  $i += 1
}
```

验证切片：

```powershell
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 ".\preencut-clips\clip_001.mp4"
```

## 13. 验证标准

任务完成后，agent 必须做这些检查：

1. `/api/agent/health` 返回 `status=ok`。
2. 任务最终 `status=completed`，`progress=1.0`。
3. `result.artifacts` 至少包含 `result.csv`、`summary_report.txt`、`transcript_report.docx`。
4. 如果开启 `enable_alignment=true`，应包含 `.srt` 字幕。
5. 下载的文件大小必须大于 0。
6. `segments` 中每条片段必须满足 `start < end`，且标签属于允许枚举。
7. 如果用户要求“派星所有提及”，不要只返回强相关；必须包含弱相关、仅提及、负面。
8. 如果用户要求“可剪正面内容”，优先返回 `强相关 + 主动讲解/正面展示`。

## 14. 常见故障

| 错误 | 原因 | 处理 |
| --- | --- | --- |
| `moov atom not found` | 上传到临时目录的视频损坏、不完整或未上传完成 | 让用户重新上传；如果源文件本身可读，可先 `ffmpeg -i input.mp4 -c copy fixed.mp4` 后再处理 |
| `CUDA error: no kernel image is available` | PyTorch 不支持当前显卡架构 | 安装匹配 CUDA/显卡的 PyTorch，或设置 `ALIGNMENT_DEVICE=cpu` |
| `OpenAI API key is not set` | `.env` 缺少所选 LLM 的 API Key | 补充 `GOOGLE_API_KEY`、`DEEPSEEK_V3_API_KEY` 等 |
| `task is not completed` | 过早下载 artifact | 先轮询到 `completed` |
| `not_found` | 服务重启后内存任务状态丢失 | 重新创建任务；已生成文件可在 `output/` 查找 |
| `result.csv` 乱码 | 打开方式不识别 UTF-8 | 当前代码写入 `utf-8-sig`；优先用 Excel 直接打开或按 UTF-8 导入 |

## 15. 返回给用户的内容

不要把完整 JSON 原样发给用户。根据用户目标组织结果：

- 简短说明处理是否完成。
- 给出保留片段数量和筛选模式。
- 列出最关键的 3-10 个片段：时间、标签、摘要。
- 附上可下载文件：`summary_report.txt`、`transcript_report.docx`、`result.csv`、字幕文件。
- 如果生成了视频切片，返回切片文件并说明筛选条件。

示例：

```text
处理完成。本次按“全部相关提及”保留 18 个派星相关片段，其中强相关 7 个、弱相关 5 个、仅提及 4 个、负面 2 个。

重点片段：
1. 00:12:30-00:13:05 强相关 / 主动讲解：讲派星 HMO 和 OPO 配方卖点。
2. 00:28:11-00:28:40 强相关 / 回答观众：回答“一段派星，二段喝什么”。
3. 00:43:02-00:44:15 弱相关 / 对比提及：与皇家美素佳儿做价格和配方对比。

已生成 result.csv、summary_report.txt、transcript_report.docx 和 srt 字幕。
```

## 16. 强制规则

1. 先健康检查，再创建任务。
2. `from-path` 只能用于 PreenCut 服务器本机存在的路径。
3. 用户要求“所有提及”时必须用 `all_mentions`，不要让 prompt 写成“只保留强相关”。
4. 用户要求“正面可剪”时，除设置 `strong_only` 外，还要本地过滤 `intent`。
5. 不要声称系统看懂了画面；当前 Agent API 没有视觉识别链路。
6. 不要在任务未完成时下载产物。
7. 对超长视频要预期耗时较长，轮询间隔建议 10-30 秒。
8. 任何下载给用户的 CSV、TXT、DOCX、SRT 都必须确认文件存在且大小大于 0。
