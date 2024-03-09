import os

from dotenv import load_dotenv
from flask import Flask
from flask_restful import Api
from gevent.pywsgi import WSGIServer

from controller import TranscriptionController

app = Flask(__name__)
api = Api(app)

api.add_resource(TranscriptionController, '/transcriptions')

if __name__ == '__main__':
    load_dotenv()
    port = int(os.environ.get('FLASK_PORT'))
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()