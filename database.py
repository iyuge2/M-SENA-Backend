from app import *
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

class Dataset(db.Model):
    __tablename__ = "Dataset"
    # db.Column(primary_key, autoincrement, default, nullable, unique, onupdate, name)
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(32), nullable=False)
    path = db.Column(db.String(128), nullable=False)
    language = db.Column(db.String(16), nullable=False, default="en")
    label_type = db.Column(db.Integer, nullable=False, default=0)
    data_params = db.Column(db.String(128), nullable=False, default="{}")
    text_format = db.Column(db.String(8), nullable=False, default='txt')
    audio_format = db.Column(db.String(8), nullable=False, default='wav')
    video_format = db.Column(db.String(8), nullable=False, default='mp4')
    has_s_label = db.Column(db.Boolean, nullable=False, default=False)
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
    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), nullable=False)
    segment_id = db.Column(db.Integer, nullable=False)
    clip_id = db.Column(db.Integer, nullable=False)
    relative_path = db.Column(db.String(128), nullable=False)
    train_mode = db.Column(db.Integer, nullable=False) # 0 - train, 1 - valid, 2 - test
    text = db.Column(db.String(256), nullable=False)
    # grade = db.Column(db.Integer, nullable=False) # 0 - labelled, 1 - simple, 2 - middle, 3 - hard
    m_label_value = db.Column(db.Float)
    m_label_by = db.Column(db.Integer) # 0 - human, 1 - machine, 2 - middle, 3 - hard
    t_label_value = db.Column(db.Float)
    t_label_by = db.Column(db.Integer) # 0 - human, 1 - machine, 2 - middle, 3 - hard
    a_label_value = db.Column(db.Float)
    a_label_by = db.Column(db.Integer) # 0 - human, 1 - machine, 2 - middle, 3 - hard
    v_label_value = db.Column(db.Float)
    v_label_by = db.Column(db.Integer) # 0 - human, 1 - machine, 2 - middle, 3 - hard

    def __repr__(self):
        return str(self.__dict__)

class Dfeature(db.Model):
    __tablename__ = "Dfeature"

    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), primary_key=True)
    feature_path = db.Column(db.String(128), nullable=False)
    description = db.Column(db.String(128))

    def __repr__(self):
        return str(self.__dict__)

class DLClassify(db.Model):
    __tablename__ = "DLClassify"

    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), primary_key=True)
    class_num = db.Column(db.Integer, nullable=False)
    init_value = db.Column(db.Integer, nullable=False)
    class_name = db.Column(db.String(128), nullable=False) # class name splitted by semicolon

    def __repr__(self):
        return str(self.__dict__)

class DLRegression(db.Model):
    __tablename__ = "DLRegression"

    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), primary_key=True)
    label_max = db.Column(db.Integer, nullable=False)
    label_min = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return str(self.__dict__)

class Model(db.Model):
    __tablename__ = "Model"
    # db.Column(primary_key, autoincrement, default, nullable, unique, onupdate, name)
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(32), nullable=False)
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
    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), nullable=False)
    model_id = db.Column(db.Integer, ForeignKey('Model.id'), nullable=False)
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
    result_id = db.Column(db.Integer, primary_key=True)
    epoch_num = db.Column(db.Integer, nullable=False)
    loss_value = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    macro_f1 = db.Column(db.Float)

    def __repr__(self):
        return str(self.__dict__)

class RgResult(db.Model):
    __tablename__ = "RgResult"
    result_id = db.Column(db.Integer, primary_key=True)
    epoch_num = db.Column(db.Integer, nullable=False)
    loss_value = db.Column(db.Float)
    mse = db.Column(db.Float)
    corr = db.Column(db.Float)

    def __repr__(self):
        return str(self.__dict__)

class ModelSave(db.Model):
    __tablename__ = "ModelSave"
    result_id = db.Column(db.Integer, primary_key=True)
    model_path = db.Column(db.String(128)) # model_name: 模型名字__数据集名字__KeyEval结果值__日期

    def __repr__(self):
        return str(self.__dict__)

class Task(db.Model):
    __tablename__ = "Task"
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_id = db.Column(db.Integer, ForeignKey('Dataset.id'), nullable=False)
    model_id = db.Column(db.Integer, ForeignKey('Model.id'), nullable=False)
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