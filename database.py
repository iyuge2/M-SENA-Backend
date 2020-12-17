import os
import argparse

from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

from constants import *

# __all__ = ['Dataset']
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
db = SQLAlchemy(app)

# class Dataset(db.Model):
#     __tablename__ = "Dataset"
#     # db.Column(primary_key, autoincrement, default, nullable, unique, onupdate, name)
#     # dataset_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     dataset_name = db.Column(db.String(32), primary_key=True, nullable=False)
#     dataset_path = db.Column(db.String(128), nullable=False)
#     language = db.Column(db.String(16), nullable=False, default="EN")
#     label_path = db.Column(db.String(32), nullable=False)
#     text_format = db.Column(db.String(8), nullable=False, default='txt')
#     audio_format = db.Column(db.String(8), nullable=False, default='wav')
#     video_format = db.Column(db.String(8), nullable=False, default='mp4')
#     raw_video_dir = db.Column(db.String(64), nullable=False)
#     audio_dir = db.Column(db.String(64), nullable=False)
#     faces_dir = db.Column(db.String(64), nullable=False)
#     has_feature = db.Column(db.Boolean, nullable=False, default=False)
#     is_locked = db.Column(db.Boolean, nullable=False, default=False)
#     description = db.Column(db.String(128))

#     def __repr__(self):
#         return str(self.__dict__)

class Dsample(db.Model):
    __tablename__ = "Dsample"
    sample_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False)
    video_id = db.Column(db.String(32), nullable=False)
    clip_id = db.Column(db.String(32), nullable=False)
    video_path = db.Column(db.String(128), nullable=False)
    text = db.Column(db.String(512), nullable=False)
    # 0 -- train, 1 -- valid, 2 -- test
    data_mode = db.Column(db.String(8), nullable=False)
    label_value = db.Column(db.Float)
    annotation = db.Column(db.String(16))
    # -1 - unlabeled, 0 - human, 1 - machine, 2 - middle, 3 - hard
    label_by = db.Column(db.Integer, nullable=False) 

    def __repr__(self):
        return str(self.__dict__)

# class Dfeature(db.Model):
#     __tablename__ = "Dfeature"

#     feature_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     dataset_name = db.Column(db.String(32), nullable=False)
#     feature_path = db.Column(db.String(128), nullable=False)
#     seq_lens = db.Column(db.String(32), nullable=False)
#     feature_dims = db.Column(db.String(32), nullable=False)
#     description = db.Column(db.String(128))

#     def __repr__(self):
#         return str(self.__dict__)

# class Model(db.Model):
#     __tablename__ = "Model"
#     # model_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     model_name = db.Column(db.String(32), primary_key=True, nullable=False)
#     args = db.Column(db.String(MAX_ARGS_LEN), nullable=False, default="{}")
#     paper_name = db.Column(db.String(128))
#     paper_url = db.Column(db.String(128))
#     description = db.Column(db.String(128))

#     def __repr__(self):
#         return str(self.__dict__)

class Result(db.Model):
    __tablename__ = "Result"
    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False)
    model_name = db.Column(db.String(32), nullable=False)
    data_mode = db.Column(db.String(8), nullable=False)
    # Tune, Normal
    is_tuning = db.Column(db.String(8), nullable=False)
    created_at = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    args = db.Column(db.String(MAX_ARGS_LEN), nullable=False, default="{}")
    save_model_path = db.Column(db.String(128))
    description = db.Column(db.String(128))

    def get_id(self):
        return str(self.result_id)

    def __repr__(self):
        return str(self.__dict__)

class EResult(db.Model):
    # results for each epoch
    # epoch_num == -1 means the final results
    __tablename__ = "EResult"
    result_id = db.Column(db.Integer, primary_key=True, nullable=False)
    epoch_num = db.Column(db.Integer, primary_key=True, nullable=False)
    loss_value = db.Column(db.String(32))
    accuracy = db.Column(db.String(32))
    f1 = db.Column(db.String(32))
    mae = db.Column(db.String(32))
    corr = db.Column(db.String(32))

    def __repr__(self):
        return str(self.__dict__)

class Task(db.Model):
    __tablename__ = "Task"
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False)
    model_name = db.Column(db.String(32), nullable=False)
    # 0 - 机器标注 1 - 模型训练，2 - 模型调参，3 - 模型测试
    task_type = db.Column(db.Integer, nullable=False)
    task_pid = db.Column(db.Integer, nullable=False)
    # 0 -- 运行中，1 -- 已完成，2 -- 运行出错 3 -- 运行终止
    state = db.Column(db.Integer, nullable=False) 
    start_time = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    end_time = db.Column(db.Date, default=datetime.utcnow() + timedelta(hours=8))
    message = db.Column(db.String(32))

    def __repr__(self):
        return str(self.__dict__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_args()
    if arg.mode == "all" or arg.mode == "drop":
        db.drop_all()
    if arg.mode == "all" or arg.mode == "create":
        db.create_all()