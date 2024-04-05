import re
import wave
from os import remove
from os.path import dirname, isfile, join, realpath, splitext
from uuid import uuid4

from flask import request
from flask_restful import Resource

from audio import Audio, millisec


class TranscriptionController(Resource):
    UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'upload')
    ALLOWED_EXTENSIONS = ['pcm', 'mp3']

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def post(self):
        def transcribe(path):
            audio = Audio(path)
            transcription = audio.transcribe()
            return transcription
        def transcribe_with_speaker(path, features):
            audio_path = path
            base_path, ext = splitext(path)
            if ext == '.pcm':
                audio_path = f'{base_path}.wav'
                with open(path, "rb") as inp_f:
                    data = inp_f.read()
                    with wave.open(audio_path, "wb") as out_f:
                        out_f.setnchannels(1)
                        out_f.setsampwidth(2) # number of bytes = 16 bits
                        out_f.setframerate(16000)
                        out_f.writeframesraw(data)
            try:
                audio = Audio(audio_path)
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
                    content = section_audio.transcribe()
                    item = {
                        'speaker': s[0].split()[-1],
                        'content': content,
                        'start': start,
                        'end': end
                    }
                    if 'embedding' in features:
                        embedding = section_audio.embed()
                        item['embedding'] = embedding.tolist()
                    response.append(item)
                return response
            except Exception as e:
                raise e
            finally:
                remove(audio_path)
                for p in section_paths:
                    remove(p)

        file = request.files['audio']
        features = []
        if 'features' in request.form:
            features = request.form['features'].replace(' ', '').split(',')
        if file and self.allowed_file(file.filename):
            _, ext = splitext(file.filename)
            filename = f'{uuid4()}{ext}'
            path = join(self.UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                if 'speaker' in features:
                    return transcribe_with_speaker(path, features)
                else:
                    return transcribe(path)
            except Exception as e:
                print(e)
                raise e
            finally:
                if isfile(path):
                    remove(path)
        return 'Bad Request'