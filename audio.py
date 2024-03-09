import numpy as np
import whisper
from pydub import AudioSegment

model = whisper.load_model("small")


class Audio:
    def __init__(self, path):
        self.path = path
    
    def transcribe(self):
        audio = AudioSegment.from_file(self.path, "pcm", sample_width=16, frame_rate=16000, channels=1)
        input = np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

        transcription = model.transcribe(input)
        return transcription