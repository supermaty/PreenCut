import faster_whisper
import json
import os
import zhconv
from datetime import datetime

from modules.speech_recognizers.speech_recognizer import SpeechRecognizer


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
    ):
        super().__init__(model_size, device, device_index=device_index,
                         compute_type=compute_type,
                         batch_size=batch_size)
        if beam_size > 0:
            self.beam_size = beam_size
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

    def transcribe(self, audio_path: str):
        """将音频文件转录为文本"""
        # 确保音频文件存在
        self.before_transcribe(audio_path)
        print(f"get audio data: {audio_path}")
        print(f"batch size = {self.batch_size}")
        audio = faster_whisper.decode_audio(audio_path)
        print("load audio success")
        segments, info = self.model.transcribe(
            audio,
            initial_prompt="请使用简体中文输出。Add punctuation after end of each line. 就比如说，我要先去吃饭。Segment at end of each sentence.",
            word_timestamps=False,
            vad_filter=True,
            beam_size=self.beam_size
        )

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

        # 把segment_list写入到 segment_list.json 文件
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'segment-data', f'segment_list_{self.model_size}_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segment_list, f, ensure_ascii=False, indent=4)
        
        # format result
        result = {"language": info.language, "segments": segment_list}
        return result
