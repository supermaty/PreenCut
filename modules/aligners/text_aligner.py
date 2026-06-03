import os
import uuid
import re
from typing import Optional, List, Dict
import torch
from modules.word_segmenter import WordSegmenter

from config import (
    ALIGNMENT_BATCH_SIZE,
    ALIGNMENT_MODEL,
    WHISPER_DEVICE,
    ALIGNMENT_DEVICE,
    ALIGNMENT_CHUNK_DURATION_SECONDS,
    TEMP_FOLDER,
)


def process_ctc_text(segments: List[Dict], language_code: str,
                     word_segmenter: Optional[object],
                     max_line_length: int) -> str:
    """处理CTC强制对齐的文本"""
    line_list = []
    for segment in segments:
        line = segment['text'].strip()
        # 如果文本长度超过最大行长度，则进行分割
        if word_segmenter and len(line) > max_line_length:
            split_lines = split_long_line(line, language_code, word_segmenter,
                                          max_line_length)
            line_list.extend(split_lines)
        else:
            line_list.append(line)
    text = ' '.join(line_list)
    return text


def split_long_line(line: str, language_code: str,
                    word_segmenter: WordSegmenter,
                    max_length: int) -> List[str]:
    output = []
    if language_code == 'zh':
        # 提取并添加特定符号中的词语到 jieba 词典
        extract_and_add_phrases(word_segmenter, line)

        line = line.strip()
        if line:
            # 1.不超过max_length
            if len(line) <= max_length:
                output.append(line)
            # 2. 超过max_length，先按标点切分
            else:
                sub_sentences = [s.strip() for s in
                                 re.split('[，。,;；：？?！…]', line)
                                 if
                                 s.strip()]
                for sentence in sub_sentences:
                    if len(sentence) <= max_length:
                        output.append(sentence)
                    # 3. 仍然超过max_line_width，使用jieba分词
                    else:
                        spit_sentences = word_segmenter.split_long_sentence(
                            sentence, max_length)
                        output.extend(spit_sentences)

    else:
        output.append(line)

    return output


def extract_and_add_phrases(word_segmenter, text):
    pattern = '(《[^》]*》|"[^"]*"|\'[^\']*\'|‘[^’]*’|“[^”]*”|\([^\)]*\)|（[^）]*）|「[^」]*」)'
    phrases = re.findall(pattern, text)
    for phrase in phrases:
        # 将词语添加到分词器的词典中
        word_segmenter.add_word(phrase)


def to_639_3(language_code: str) -> str:
    """将语言代码转换为ISO 639-3格式"""
    language_code = language_code.lower()
    if language_code == 'zh':
        return 'cmn'  # 中文
    elif language_code == 'en':
        return 'eng'  # 英语
    elif language_code == 'es':
        return 'spa'  # 西班牙语
    elif language_code == 'fr':
        return 'fra'  # 法语
    elif language_code == 'de':
        return 'deu'  # 德语
    elif language_code == 'ja':
        return 'jpn'
    elif language_code == 'ko':
        return 'kor'
    else:
        return "eng"  # 默认返回英文代码


class TextAligner:
    """文本对齐器类，用于将文本与音频对齐"""

    def __init__(self, language_code: Optional[str] = None,
                 word_segmenter: Optional[object] = None,
                 max_line_length: int = 16):
        self.language_code = language_code or 'zh'  # 默认语言代码为中文
        self.model = self._load_model()
        self.word_segmenter = word_segmenter
        self.max_line_length = max_line_length

    def _load_model(self):
        if ALIGNMENT_MODEL == 'whisperx':
            try:
                import whisperx
                print(
                    f"加载WhisperX对齐模型，语言{self.language_code}，设备{WHISPER_DEVICE}")
                model = whisperx.load_align_model(
                    language_code=self.language_code, device=WHISPER_DEVICE)
                return model
            except ImportError:
                raise ImportError(
                    "WhisperX not installed. Please install with 'pip install whisperx'")
        elif ALIGNMENT_MODEL == 'ctc-forced-aligner':
            try:
                from ctc_forced_aligner import load_alignment_model
                print(
                    f"加载CTC强制对齐模型，语言{self.language_code}，设备{ALIGNMENT_DEVICE}")
                model = load_alignment_model(
                    ALIGNMENT_DEVICE,
                    dtype=torch.float16 if ALIGNMENT_DEVICE == "cuda" else torch.float32,
                )
                return model
            except ImportError:
                raise ImportError(
                    "CTC Forced Aligner not installed. Please install with 'pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git'")
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")

    def _run_ctc_alignment(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """对单段音频运行 CTC 对齐，返回带 start/end/text 的 segments 列表。"""
        from ctc_forced_aligner import (
            generate_emissions,
            preprocess_text,
            get_alignments,
            get_spans,
            postprocess_results,
        )
        import soundfile as sf

        alignment_model, alignment_tokenizer = self.model
        waveform, sample_rate = sf.read(audio_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        elif waveform.ndim > 2:
            waveform = waveform.flatten()
        assert waveform.ndim == 1
        audio_waveform = torch.from_numpy(waveform).to(
            dtype=alignment_model.dtype,
            device=alignment_model.device
        )
        text = process_ctc_text(segments, self.language_code,
                                self.word_segmenter, self.max_line_length)
        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=ALIGNMENT_BATCH_SIZE
        )
        code_639_3 = to_639_3(self.language_code)
        tokens_starred, text_starred = preprocess_text(
            text, romanize=True, language=code_639_3,
        )
        segs, scores, blank_token = get_alignments(
            emissions, tokens_starred, alignment_tokenizer,
        )
        spans = get_spans(tokens_starred, segs, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def align(self, segments: List[Dict], audio_path: str) -> str:
        """将文本与音频对齐"""
        if ALIGNMENT_MODEL == 'whisperx':
            # 使用WhisperX进行对齐
            import whisperx
            audio = whisperx.load_audio(audio_path)
            align_model, align_model_metadata = self.model
            result = whisperx.align(segments, align_model, align_model_metadata,
                                    audio, WHISPER_DEVICE,
                                    return_char_alignments=False)
            return result
        elif ALIGNMENT_MODEL == 'ctc-forced-aligner':
            if self.language_code != 'zh':
                return {"segments": segments}
            from utils import get_media_duration
            from modules.video_processor import VideoProcessor

            duration = get_media_duration(audio_path)
            if duration is None or duration <= ALIGNMENT_CHUNK_DURATION_SECONDS:
                # 短音频：整段对齐
                result = self._run_ctc_alignment(audio_path, segments)
                return {"segments": result}

            # 长音频：按固定时长切分后分批对齐，避免 MemoryError
            step_sec = ALIGNMENT_CHUNK_DURATION_SECONDS
            num_chunks = max(1, int((duration + step_sec - 1e-6) // step_sec))
            print(f"长音频分片对齐: 总长 {duration/60:.1f} 分钟，每片最多 {step_sec/60:.0f} 分钟，共 {num_chunks} 片")
            temp_dir = os.path.join(TEMP_FOLDER, "align_chunks", str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            all_segments = []
            t_start = 0.0
            chunk_index = 0
            try:
                while t_start < duration:
                    t_end = min(t_start + step_sec, duration)
                    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}.wav")
                    VideoProcessor.extract_audio_segment(
                        audio_path, t_start, t_end, chunk_path
                    )
                    segs_in_chunk = [
                        s for s in segments
                        if s["end"] > t_start and s["start"] < t_end
                    ]
                    segs_rel = [
                        {
                            "start": max(0.0, s["start"] - t_start),
                            "end": min(t_end - t_start, s["end"] - t_start),
                            "text": s["text"],
                        }
                        for s in segs_in_chunk
                    ]
                    if segs_rel:
                        aligned = self._run_ctc_alignment(chunk_path, segs_rel)
                        for seg in aligned:
                            seg["start"] += t_start
                            seg["end"] += t_start
                            all_segments.append(seg)
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
                    t_start = t_end
                    chunk_index += 1
                all_segments.sort(key=lambda s: (s["start"], s["end"]))
                return {"segments": all_segments}
            finally:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")
