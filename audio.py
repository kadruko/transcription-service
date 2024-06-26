import os
from pathlib import Path

import numpy as np
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s


load_dotenv()

model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HUGGINGFACE_ACCESS_TOKEN'])
encoder = VoiceEncoder()


class Audio:
    def __init__(self, path):
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.base_path = path.split('.')[0]
        self.format = path.split('.')[-1]
        if self.format == 'pcm':
            self.audio = AudioSegment.from_file(self.path, self.format, sample_width=16, frame_rate=16000, channels=1)
        else:
            self.audio = AudioSegment.from_file(self.path, self.format)

    def transcribe(self):
        input = self.path
        if self.format == 'pcm':
            input = np.frombuffer(self.audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

        transcription = model.transcribe(input)
        return transcription
    
    def embed(self):
        path = Path(self.path)
        wav = preprocess_wav(path)
        embedding = encoder.embed_utterance(wav)
        return embedding
    
    def diarize_speaker(self):
        def match_target_amplitude(sound, target_dBFS):
            change_in_dBFS = target_dBFS - sound.dBFS
            return sound.apply_gain(change_in_dBFS)

        dz_audio_path = f'{self.base_path}-dz.{self.format}'
        dz_audio = match_target_amplitude(self.audio, -20.0)
        dz_audio.export(dz_audio_path, format=self.format)

        try:
            result = pipeline({ 'audio': dz_audio_path })
        except Exception as e:
            raise e
        finally:
            os.remove(dz_audio_path)

        dz_path = f'{self.base_path}-dz.txt'
        with open(dz_path, "w") as text_file:
            text_file.write(str(result))
        
        try:
            dzs = open(dz_path).read().splitlines()
        except Exception as e:
            raise e
        finally:
            os.remove(dz_path)

        groups = []
        group = []
        for dz in dzs:
            if len(group) == 0 or group[0].split()[-1] == dz.split()[-1]:
                group.append(dz)
            else:
                groups.append(group)
                group = [dz]
        groups.append(group)

        return groups
