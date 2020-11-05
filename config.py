import os

class BaseConfig(object):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    JSON_AS_ASCII = False # Chinese
    DATABASE_URL = "mysql://iyuge2:960606saandsb@localhost/sena"
    
    TEXT_FORMAT = ['csv']
    AUDIO_FORMAT = ['wav', 'mp3']
    VIDEO_FORMAT = ['mp4', 'avi', 'flv']


class TestConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False