
import logging

from config.constants import APP_SETTINGS
from flask import Flask
from flask_sock import Sock
from flask_sqlalchemy import SQLAlchemy
from multiprocessing import Queue


logger = logging.getLogger('app')

app = Flask(__name__)
app.config.from_object(APP_SETTINGS)

sockets = Sock(app)

db = SQLAlchemy(app)
db.create_all()

progress_queue = Queue(maxsize=512)
result_queue = Queue(maxsize=512)

from .analysis import *
from .data import *
from .feature import *
from .labeling import *
from .model import *
from .settings import *
from .task import *
from .user import *
