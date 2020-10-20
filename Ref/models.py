from app import *
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta


class User(db.Model):
    __tablename__ = "User"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), default="000000")
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    updated_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8),
                           onupdate=datetime.utcnow() + timedelta(hours=8))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if kwargs.get('password'):
            self.password = bcrypt.generate_password_hash(kwargs.get('password')).decode('utf-8')
        else:
            self.password = bcrypt.generate_password_hash("000000").decode('utf-8')

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return str(self.__dict__)


class Project(db.Model):
    __tablename__ = "Project"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('User.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.String(256))
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    updated_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8),
                           onupdate=datetime.utcnow() + timedelta(hours=8))

    def __repr__(self):
        return str(self.__dict__)


class Data(db.Model):
    __tablename__ = "Data"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, ForeignKey('Project.id'), nullable=False)
    name_original = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    path = db.Column(db.String(128))
    dtype = db.Column(db.String(16))
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))

    def __repr__(self):
        return str(self.__dict__)


class DataSeg(db.Model):
    __tablename__ = "DataSeg"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, ForeignKey('Project.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    start_time = db.Column(db.FLOAT())
    stop_time = db.Column(db.FLOAT())
    duration = db.Column(db.FLOAT())
    path = db.Column(db.String(128))
    dtype = db.Column(db.String(16))
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))

    def __repr__(self):
        return str(self.__dict__)


class WordCloud(db.Model):
    __tablename__ = "WordCloud"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, ForeignKey('Project.id'), nullable=False)
    path = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))

    def __repr__(self):
        return str(self.__dict__)