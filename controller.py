import re
import wave
from os import remove
from os.path import dirname, join, realpath
from uuid import uuid4

from flask import request
from flask_restful import Resource

from audio import Audio, millisec


class TranscriptionController(Resource):
    UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'upload')
    ALLOWED_EXTENSIONS = ['pcm']

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def post(self):
        def transcribe(path):
            audio = Audio(path)
            transcription = audio.transcribe()
            return transcription
        def transcribe_with_speaker(path):
            base_path = path.split('.pcm')[0]
            wav_path = f'{base_path}.wav'
            with open(path, "rb") as inp_f:
                data = inp_f.read()
                with wave.open(wav_path, "wb") as out_f:
                    out_f.setnchannels(1)
                    out_f.setsampwidth(2) # number of bytes = 16 bits
                    out_f.setframerate(16000)
                    out_f.writeframesraw(data)
            try:
                audio = Audio(wav_path)
                sections = audio.diarize_speaker()
                gidx = -1
                response = []
                section_paths = []
                for s in sections:
                    start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=s[0])[0]
                    end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=s[-1])[1]
                    start = millisec(start) #- spacermilli
                    end = millisec(end)  #- spacermilli
                    print(start, end)
                    gidx += 1
                    section_path = f'{base_path}-{gidx}.wav'
                    audio.audio[start:end].export(section_path, format='wav')
                    section_paths.append(section_path)
                    section_audio = Audio(section_path)
                    transcription = section_audio.transcribe()
                    response.append({
                        'speaker': s[0].split()[-1],
                        'transcription': transcription,
                        'start': start,
                        'end': end
                    })
                return response
            except Exception as e:
                raise e
            finally:
                remove(wav_path)
                for p in section_paths:
                    remove(p)

        file = request.files['audio']
        features = []
        if 'features' in request.form:
            features = request.form['features'].replace(' ', '').split(',')
        if file and self.allowed_file(file.filename):
            filename = f'{uuid4()}.pcm'
            path = join(self.UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                if 'speaker-diarization' in features:
                    return transcribe_with_speaker(path)
                else:
                    return transcribe(path)
            except Exception as e:
                print(e)
                raise e
            finally:
                remove(path)
        return 'Bad Request'