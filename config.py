import os
from constants import DATABASE_URL

class BaseConfig(object):
    DEBUG = False
    # SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    JSON_AS_ASCII = False # Chinese
    SECRET_KEY = 'Iod3TLUOmU6FSBGJeca7-Q'

class TestConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False