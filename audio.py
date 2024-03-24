import os
import re

import numpy as np
import whisper
from pyannote.audio import Inference, Model, Pipeline
from pydub import AudioSegment


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HUGGINGFACE_ACCESS_TOKEN'])
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.environ['HUGGINGFACE_ACCESS_TOKEN'])
inference = Inference(embedding_model, window='whole')


class Audio:
    def __init__(self, path):
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.base_path = path.split('.')[0]
        self.format = path.split('.')[-1]
        self.audio = AudioSegment.from_file(self.path, self.format, sample_width=16, frame_rate=16000, channels=1)

    def transcribe(self):
        input = np.frombuffer(self.audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

        transcription = model.transcribe(input)
        return transcription
    
    def embed(self):
        embedding = inference(self.path)
        return embedding
    
    def diarize_speaker(self):
        result = pipeline({ 'audio': self.path })
        dz_path = f'{self.base_path}-dz.txt'

        with open(dz_path, "w") as text_file:
            text_file.write(str(result))
        
        try:
            dzs = open(dz_path).read().splitlines()
        except Exception as e:
            raise e
        finally:
            pass
            os.remove(dz_path)

        # Grouping
        groups = []
        g = []
        lastend = 0

        for d in dzs:   
            if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
                groups.append(g)
                g = []
        
        g.append(d)
        
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if lastend > end:       #segment engulfed by a previous segment
            groups.append(g)
            g = [] 
        else:
            lastend = end
        if g:
            groups.append(g)
        
        return groups