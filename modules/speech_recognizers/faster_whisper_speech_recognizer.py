import faster_whisper
import zhconv

from modules.speech_recognizers.speech_recognizer import SpeechRecognizer
from config import (
    FASTER_WHISPER_USE_BATCHED_PIPELINE,
    WHISPER_INITIAL_PROMPT,
    VAD_MIN_SILENCE_DURATION_MS,
)


class FasterWhisperSpeechRecognizer(SpeechRecognizer):
    beam_size = 5

    def __init__(
            self,
            model_size,
            device,
            device_index,
            compute_type,
            batch_size=16,
            beam_size=5,
            language=None,
    ):
        super().__init__(model_size, device, device_index=device_index,
                         compute_type=compute_type,
                         batch_size=batch_size)
        if beam_size > 0:
            self.beam_size = beam_size
        self.language = language
        print(f"加载Whisper模型: {self.model_size}")
        print(f"device = {self.device}")
        print(f"{self.model_size, self.device, self.compute_type, self.opts}")
        if self.device == 'cpu':
            self.model = faster_whisper.WhisperModel(self.model_size,
                                                     device=self.device,
                                                     compute_type=self.compute_type)
        else:
            self.model = faster_whisper.WhisperModel(self.model_size,
                                                     device=self.device,
                                                     device_index=self.device_index,
                                                     compute_type=self.compute_type)
        self.batched_pipeline = None
        if (
            FASTER_WHISPER_USE_BATCHED_PIPELINE
            and hasattr(faster_whisper, "BatchedInferencePipeline")
        ):
            self.batched_pipeline = faster_whisper.BatchedInferencePipeline(
                model=self.model
            )
        print(f"batched pipeline = {self.batched_pipeline is not None}")

    def transcribe(self, audio_path: str):
        """将音频文件转录为文本"""
        # 确保音频文件存在
        self.before_transcribe(audio_path)
        print(f"get audio data: {audio_path}")
        print(f"batch size = {self.batch_size}")
        audio = faster_whisper.decode_audio(audio_path)
        print("load audio success")
        kwargs = dict(
            word_timestamps=False,
            vad_filter=True,
            beam_size=self.beam_size,
        )
        if self.batched_pipeline is not None:
            kwargs["batch_size"] = self.batch_size
            kwargs["without_timestamps"] = False
        if WHISPER_INITIAL_PROMPT:
            kwargs["initial_prompt"] = WHISPER_INITIAL_PROMPT
        if self.language is not None:
            kwargs["language"] = self.language
        if VAD_MIN_SILENCE_DURATION_MS is not None:
            kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            )
        transcriber = self.batched_pipeline or self.model
        segments, info = transcriber.transcribe(audio, **kwargs)

        segment_list = []

        # 读取 segment_list.json 文件
        # json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'segment-data', 'segment_list_large-v3-turbo_20251226190905.json')
        # with open(json_path, 'r', encoding='utf-8') as f:
        #     segment_list = json.load(f)
        
        for segment in segments:
            simplified_text = zhconv.convert(segment.text, 'zh-cn')
            segment_list.append({
                'start': float(f'{segment.start:.2f}'),
                'end': float(f'{segment.end:.2f}'),
                'text': simplified_text
            })

        # 断句文件改为在 processing_queue 中写入（纠错+对齐后的版本），此处不再写原始 ASR 结果
        # format result
        result = {"language": info.language, "segments": segment_list}
        return result
