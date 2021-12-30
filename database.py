import argparse
import os
import shutil
from datetime import datetime, timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, PrimaryKeyConstraint
from sqlalchemy.sql.schema import ForeignKey

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
    data_mode = db.Column(db.String(8), index=True) # 0 -- train, 1 -- valid, 2 -- test
    label_value = db.Column(db.Float, index=True) # regression label
    annotation = db.Column(db.String(16), index=True, default='-') # class label in string
    label_T = db.Column(db.Float) # text regression label
    label_A = db.Column(db.Float) # audio regression label
    label_V = db.Column(db.Float) # video regression label

    __table_args__ = (db.UniqueConstraint('dataset_name', 'video_id', 'clip_id', name='ix_dataset_video_clip'),)

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
        db.DateTime, default=datetime.now(), index=True)
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


class SResults(db.Model):
    # results for each sample
    __tablename__ = "SResults"
    result_id = db.Column(db.Integer, nullable=False)
    sample_id = db.Column(db.Integer, nullable=False)
    label_value = db.Column(db.String(16), nullable=False)
    predict_value = db.Column(db.String(16), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("result_id", "sample_id"),
    )

    def __repr__(self):
        return str(self.__dict__)


class EResult(db.Model):
    # results for each epoch
    __tablename__ = "EResult"
    result_id = db.Column(db.Integer, nullable=False)
    epoch_num = db.Column(db.Integer, nullable=False)
    results = db.Column(db.String(256), nullable=False)
    # json {"train": {"loss": ***, "accuracy": ***, "f1": ***}, "valid": {***}}
    __table_args__ = (
        PrimaryKeyConstraint("result_id", "epoch_num"),
    )

    def __repr__(self):
        return str(self.__dict__)


class Task(db.Model):
    __tablename__ = "Task"
    task_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False, index=True)
    model_name = db.Column(db.String(32), nullable=False, index=True)
    task_type = db.Column(db.Integer, nullable=False, index=True) # 0 - 机器标注 1 - 模型训练，2 - 模型调参，3 - 模型测试，4 - 特征抽取
    task_pid = db.Column(db.Integer, nullable=False, index=True)
    state = db.Column(db.Integer, nullable=False, index=True) # 0 -- 运行中，1 -- 已完成，2 -- 运行出错 3 -- 运行终止
    start_time = db.Column(
        db.DateTime, default=datetime.now(), index=True)
    end_time = db.Column(
        db.DateTime, default=datetime.now())
    message = db.Column(db.String(32))

    def __repr__(self):
        return str(self.__dict__)


class User(db.Model):
    __tablename__ = "User"
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_name = db.Column(db.String(64), nullable=False, index=True, unique=True)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=True, server_default='0')

    def __repr__(self):
        return str(self.__dict__)


class Annotation(db.Model):
    __tablename__ = "Annotation"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_name = db.Column(db.String(64), ForeignKey("User.user_name"), nullable=False)
    dataset_name = db.Column(db.String(32), nullable=False)
    video_id = db.Column(db.String(32), nullable=False)
    clip_id = db.Column(db.String(32), nullable=False)
    label_M = db.Column(db.Float)
    label_T = db.Column(db.Float)
    label_A = db.Column(db.Float)
    label_V = db.Column(db.Float)

    __table_args__ = (db.ForeignKeyConstraint(['dataset_name', 'video_id', 'clip_id'],
                                              ['Dsample.dataset_name', 'Dsample.video_id', 'Dsample.clip_id']),
                      db.UniqueConstraint('user_name', 'dataset_name', 'video_id', 'clip_id', name='ix_user_dataset_video_clip',),
                      db.Index('ix_user_dataset_labelM', 'user_name', 'dataset_name', 'label_M'),
                      db.Index('ix_user_dataset_labelT', 'user_name', 'dataset_name', 'label_T'),
                      db.Index('ix_user_dataset_labelA', 'user_name', 'dataset_name', 'label_A'),
                      db.Index('ix_user_dataset_labelV', 'user_name', 'dataset_name', 'label_V'),
                     )

    def __repr__(self):
        return str(self.__dict__)


class Feature(db.Model):
    __tablename__ = "Feature"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset_name = db.Column(db.String(32), nullable=False)
    feature_name = db.Column(db.String(32), nullable=False)
    feature_T = db.Column(db.String(128))
    feature_A = db.Column(db.String(128))
    feature_V = db.Column(db.String(128))
    feature_path = db.Column(db.String(256), nullable=False)
    description = db.Column(db.String(256))

    __table_args__ = (
        db.UniqueConstraint('dataset_name', 'feature_name', 'feature_T', 'feature_A', 'feature_V', name='ix_dataset_feature_T_A_V',),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='create')
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
        # db.session.execute("""
        #     INSERT INTO `User` (`user_name`, `password`, `is_admin`) VALUES ('admin', SHA1('m-sena'), 1);
        # """)
        db.session.commit()
