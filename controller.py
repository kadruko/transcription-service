from os import remove
from os.path import dirname, join, realpath
from uuid import uuid4

from flask import request
from flask_restful import Resource

from audio import Audio


class TranscriptionController(Resource):
    UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'upload')
    ALLOWED_EXTENSIONS = ['pcm']

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def post(self):
        file = request.files['audio']
        if file and self.allowed_file(file.filename):
            filename = f'{uuid4()}.pcm'
            path = join(self.UPLOAD_FOLDER, filename)
            file.save(path)
            try:
                audio = Audio(path)
                transcription = audio.transcribe()
                return transcription
            except Exception as e:
                print(e)
                pass
            finally:
                remove(path)
        return 'Bad Request'