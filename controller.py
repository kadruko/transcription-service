import re
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
            audio = Audio(path)
            sections = audio.diarize_speaker()
            gidx = -1
            response = []
            for s in sections:
                start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=s[0])[0]
                end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=s[-1])[1]
                start = millisec(start) #- spacermilli
                end = millisec(end)  #- spacermilli
                print(start, end)
                gidx += 1
                base_path = path.split('.pcm')[0]
                section_path = f'{base_path}-{gidx}.pcm'
                audio[start:end].export(section_path, format='pcm')
                section_audio = Audio(section_path)
                transcription = section_audio.transcribe()
                response.append({
                    'speaker': s[0].split()[-1],
                    'transcription': transcription
                })
            return sections

        file = request.files['audio']
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
                pass
            finally:
                remove(path)
        return 'Bad Request'