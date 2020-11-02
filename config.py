import os

class BaseConfig(object):
    DATABASE_URL = "mysql://iyuge2:960606saandsb@localhost/sena"
    SUPPORT_FORMAT = {
        'text': ['txt'],
        'audio': ['wav', 'mp3'],
        'video': ['mp4', 'avi', 'flv']
    }


# default config
# class BaseConfig(object):
#     DEBUG = False
#     # shortened for readability
#     SECRET_KEY = '\xbf\xb0\x11\xb1\xcd\xf9\xba\x8bp\x0c...'
#     SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
#     SQLALCHEMY_TRACK_MODIFICATIONS = True
#     JSON_AS_ASCII = False # Chinese
#     # Upload
#     MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300 MB limit
#     UPLOADED_FILE_DEST = os.path.join(os.getcwd(), 'input')
#     OUTPUT_FILE_DEST = os.path.join(os.getcwd(), 'output')
#     print(SQLALCHEMY_DATABASE_URI)


# class TestConfig(BaseConfig):
#     DEBUG = True
#     TESTING = True
#     WTF_CSRF_ENABLED = False
#     SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# class DevelopmentConfig(BaseConfig):
#     DEBUG = True


# class ProductionConfig(BaseConfig):
#     DEBUG = False
