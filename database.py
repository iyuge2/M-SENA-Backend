import os

from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

# __all__ = ['Dataset']
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
db = SQLAlchemy(app)

class Dataset(db.Model):
    __tablename__ = "Dataset"
    # db.Column(primary_key, autoincrement, default, nullable, unique, onupdate, name)
    name = db.Column(db.String(32), primary_key=True, nullable=False)
    path = db.Column(db.String(128), nullable=False)
    audio_dir = db.Column(db.String(64), nullable=False)
    faces_dir = db.Column(db.String(64), nullable=False)
    label_path = db.Column(db.String(64), nullable=False)
    language = db.Column(db.String(16), nullable=False, default="en")
    label_type = db.Column(db.String(16), nullable=False, default="classification")
    text_format = db.Column(db.String(8), nullable=False, default='txt')
    audio_format = db.Column(db.String(8), nullable=False, default='wav')
    video_format = db.Column(db.String(8), nullable=False, default='mp4')
    has_feature = db.Column(db.Boolean, nullable=False, default=False)
    is_locked = db.Column(db.Boolean, nullable=False, default=False)
    description = db.Column(db.String(128))

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return str(self.__dict__)


class Dsample(db.Model):
    __tablename__ = "Dsample"
    sample_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False)
    video_id = db.Column(db.String(32), nullable=False)
    clip_id = db.Column(db.String(32), nullable=False)
    video_path = db.Column(db.String(128), nullable=False)
    text = db.Column(db.String(256), nullable=False)
    sample_mode = db.Column(db.String(8), nullable=False) # train / valid / test
    label_value = db.Column(db.Float)
    label_by = db.Column(db.Integer, nullable=False) # -1 - unlabelled, 0 - human, 1 - machine, 2 - middle, 3 - hard

    def __repr__(self):
        return str(self.__dict__)

class Dfeature(db.Model):
    __tablename__ = "Dfeature"

    dataset_name = db.Column(db.String(32), primary_key=True, nullable=False)
    feature_path = db.Column(db.String(128), nullable=False)
    input_lens = db.Column(db.String(32), nullable=False)
    feature_dims = db.Column(db.String(32), nullable=False)
    description = db.Column(db.String(128))

    def __repr__(self):
        return str(self.__dict__)

class Model(db.Model):
    __tablename__ = "Model"
    # db.Column(primary_key, autoincrement, default, nullable, unique, onupdate, name)
    name = db.Column(db.String(32), primary_key=True, nullable=False)
    common_params = db.Column(db.String(256), nullable=False, default="{}")
    specific_params = db.Column(db.String(256), nullable=False, default="{}")
    description = db.Column(db.String(128))

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return str(self.__dict__)

class Result(db.Model):
    __tablename__ = "Result"
    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), ForeignKey('Dataset.name'), nullable=False)
    model_name = db.Column(db.String(32), ForeignKey('Model.name'), nullable=False)
    is_tuning = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    params = db.Column(db.String(1024), nullable=False, default="{}")
    key_eval_name = db.Column(db.String(16), nullable=False)
    key_eval_value = db.Column(db.Float, nullable=False)

    def get_id(self):
        return str(self.result_id)

    def __repr__(self):
        return str(self.__dict__)

class CLResult(db.Model):
    __tablename__ = "CLResult"
    result_id = db.Column(db.Integer, primary_key=True, nullable=True)
    epoch_num = db.Column(db.Integer, nullable=False)
    loss_value = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    macro_f1 = db.Column(db.Float)

    def __repr__(self):
        return str(self.__dict__)

class RgResult(db.Model):
    __tablename__ = "RgResult"
    result_id = db.Column(db.Integer, primary_key=True, nullable=True)
    epoch_num = db.Column(db.Integer, nullable=False)
    loss_value = db.Column(db.Float)
    mse = db.Column(db.Float)
    corr = db.Column(db.Float)

    def __repr__(self):
        return str(self.__dict__)

class ModelSave(db.Model):
    __tablename__ = "ModelSave"
    result_id = db.Column(db.Integer, primary_key=True, nullable=True)
    model_path = db.Column(db.String(128)) # model_name: 模型名字__数据集名字__KeyEval结果值__日期

    def __repr__(self):
        return str(self.__dict__)

class Task(db.Model):
    __tablename__ = "Task"
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), ForeignKey('Dataset.name'), nullable=False)
    model_name = db.Column(db.String(32), ForeignKey('Model.name'), nullable=False)
    # 0 - 主动学习预训练，1 - 特征预处理，2 - 模型训练，3 - 模型调参，4 - 模型测试
    task_type = db.Column(db.Integer, nullable=False)
    pid = db.Column(db.Integer, nullable=False)
    state = db.Column(db.Integer, nullable=False) # 0 -- 运行中，1 -- 已完成，2 -- 出错
    start_time = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    end_time = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    error_info = db.Column(db.String(128))

    def __repr__(self):
        return str(self.__dict__)

if __name__ == '__main__':
    db.drop_all()
    db.create_all()