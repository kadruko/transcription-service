import os
import re

import numpy as np
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=os.environ['HUGGINGFACE_TOKEN'])


class Audio:
    def __init__(self, path):
        self.path = path
        self.audio = AudioSegment.from_file(self.path, "pcm", sample_width=16, frame_rate=16000, channels=1)

    def transcribe(self):
        input = np.frombuffer(self.audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

        transcription = model.transcribe(input)
        return transcription
    
    def diarize_speaker(self):
        dzs = pipeline(self.path)

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