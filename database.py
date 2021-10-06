import argparse
import os
import shutil
from datetime import datetime, timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index

from constants import *

app = Flask(__name__)
app.config.from_object(APP_SETTINGS)
db = SQLAlchemy(app)


class Dsample(db.Model):
    # Sample Details
    __tablename__ = "Dsample"
    sample_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False, index=True)
    video_id = db.Column(db.String(32), nullable=False, index=True)
    clip_id = db.Column(db.String(32), nullable=False, index=True)
    video_path = db.Column(db.String(128), nullable=False)
    text = db.Column(db.String(SQL_MAX_TEXT_LEN), nullable=False)
    data_mode = db.Column(db.String(8), nullable=False, index=True) # 0 -- train, 1 -- valid, 2 -- test
    label_value = db.Column(db.Float, index=True)
    annotation = db.Column(db.String(16), index=True)

    def __repr__(self):
        return str(self.__dict__)


class Result(db.Model):
    __tablename__ = "Result"
    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False, index=True)
    model_name = db.Column(db.String(32), nullable=False, index=True)
    # Tune, Normal
    is_tuning = db.Column(db.String(8), nullable=False, index=True)
    created_at = db.Column(
        db.DateTime, default=datetime.utcnow() + timedelta(hours=8), index=True)
    args = db.Column(db.String(MAX_ARGS_LEN), nullable=False, default="{}")
    save_model_path = db.Column(db.String(128))
    # final test results
    loss_value = db.Column(db.Float, nullable=False, index=True)
    accuracy = db.Column(db.Float, nullable=False, index=True)
    f1 = db.Column(db.Float, nullable=False, index=True)
    description = db.Column(db.String(128))

    def get_id(self):
        return str(self.result_id)

    def __repr__(self):
        return str(self.__dict__)


# class SResults(db.Model):
#     __tablename__ = "SResults"
#     result_id = db.Column(db.Integer, primary_key=True, nullable=False)
#     sample_id = db.Column(db.Integer, index=True, nullable=False)
#     label_value = db.Column(db.String(16), nullable=False)
#     predict_value = db.Column(db.String(16), nullable=False)

#     def __repr__(self):
#         return str(self.__dict__)


class EResult(db.Model):
    # results for each epoch
    __tablename__ = "EResult"
    result_id = db.Column(db.Integer, primary_key=True, nullable=False)
    epoch_num = db.Column(db.Integer, nullable=False)
    results = db.Column(db.String(256), nullable=False)
    # json {"train": {"loss": ***, "accuracy": ***, "f1": ***}, "valid": {***}}
    __table_args__ = (Index('id_epoch', "result_id", "epoch_num"), )

    def __repr__(self):
        return str(self.__dict__)


class Task(db.Model):
    __tablename__ = "Task"
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False, index=True)
    model_name = db.Column(db.String(32), nullable=False, index=True)
    task_type = db.Column(db.Integer, nullable=False, index=True) # 0 - 机器标注 1 - 模型训练，2 - 模型调参，3 - 模型测试
    task_pid = db.Column(db.Integer, nullable=False, index=True)
    state = db.Column(db.Integer, nullable=False, index=True) # 0 -- 运行中，1 -- 已完成，2 -- 运行出错 3 -- 运行终止
    start_time = db.Column(
        db.DateTime, default=datetime.utcnow() + timedelta(hours=8), index=True)
    end_time = db.Column(
        db.DateTime, default=datetime.utcnow() + timedelta(hours=8))
    message = db.Column(db.String(32))

    def __repr__(self):
        return str(self.__dict__)


class User(db.Model):
    __tablename__ = "User"
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_name = db.Column(db.String(64), nullable=False, index=True, unique=True)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    if arg.mode == "all" or arg.mode == "drop":
        # clear tmp dir
        if os.path.exists(MODEL_TMP_SAVE):
            shutil.rmtree(MODEL_TMP_SAVE)
        db.drop_all()
    if arg.mode == "all" or arg.mode == "create":
        db.create_all()
