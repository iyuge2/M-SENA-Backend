from .constants import DATABASE_URL

class BaseConfig(object):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    JSON_AS_ASCII = False # Chinese
    SECRET_KEY = '7e6c8dcc8e1da2bd06946ec688de9553'

# class TestConfig(BaseConfig):
#     DEBUG = True
#     TESTING = True
#     WTF_CSRF_ENABLED = False
#     SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class DevelopmentConfig(BaseConfig):
    DEBUG = True

class ProductionConfig(BaseConfig):
    DEBUG = False