import os
import json
import xlwt
import subprocess

from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import load_only
from flask_cors import CORS
from flask_uploads import UploadSet, configure_uploads

from config import BaseConfig
from functions import strBool

# session.clear()
app = Flask(__name__)
# support cross-domain access
CORS(app, supports_credentials=True)

app.config['SQLALCHEMY_DATABASE_URI'] = BaseConfig.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True

db = SQLAlchemy(app)

# file = UploadSet('file', VIDEOS+AUDIOS+TEXT)
# configure_uploads(app, file)

from database import *

@app.route('/')
def hello_world():
    return "Welcome to M-SENA Platform!"

@app.route('/data/create_dataset', methods=['POST'])
def create_dataset():
    fields = ['name', 'path', 'language', 'label_type', 'data_params', 'text_format', \
                'audio_format', 'video_format', 'has_s_label', 'has_feature', 'is_locked']

    if not all(field in request.form for field in fields):
        return "Missing parameter(s)"
    
    # check model name
    names = Dataset.query.filter_by(name=request.form['name']).first()
    if names:
        return request.form['name'] + " has existed!"

    # check data format
    for d_type in ['text', 'audio', 'video']:
        if request.form[d_type + '_format'] not in BaseConfig.SUPPORT_FORMAT[d_type]:
            return "Error format in " + d_type + '. Only ' + \
                   '/'.join(BaseConfig.SUPPORT_FORMAT[d_type]) + ' support'
    # TODO: data path check && scan database

    # add new dataset to database
    payload = Dataset(
        name=request.form['name'],
        path=request.form['path'],
        language=request.form['language'],
        label_type=int(request.form['label_type']),
        data_params=request.form['data_params'],
        text_format=request.form['text_format'],
        audio_format=request.form['audio_format'],
        video_format=request.form['video_format'],
        has_s_label=strBool(request.form['has_s_label']),
        has_feature=strBool(request.form['has_feature']),
        is_locked=strBool(request.form['is_locked']),
        description=request.form['description'] if 'description' in request.form else ""
    )

    db.session.add(payload)
    db.session.commit()

    return "success"

@app.route('/data/info', methods=['GET'])
def get_datasets_info():
    datasets = Dataset.query.all()
    res = {}
    for dataset in datasets:
        p = dataset.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res[str(p['id'])] = p
    return str(res).replace("\'", "\"")


@app.route('/data/<dataset_id>/details', methods=['GET'])
def get_dataset_details(dataset_id):
    dataset_id = int(dataset_id)
    datasets = Dataset.query.filter_by(id=dataset_id).first()
    res = {}
    if datasets:
        p = datasets.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res["base_info"] = p
        # details
        samples = Dsample.query.filter_by(dataset_id=id).all()
        sample_res = {}
        for sample in samples:
            p = sample.__dict__.copy()
            p.pop('_sa_instance_state', None)
            sample_res[str(p['segment_id']) + "_" + str(p['clip_id'])] = p
        res["detail_info"] = sample_res
    else:
        return str(dataset_id) + " is not existed!"

    return str(res).replace("\'", "\"")

@app.route('/data/<dataset_id>/<segment_id>/<clip_id>', methods=['GET'])
def get_clip_video(dataset_id, segment_id, clip_id):
    fields = ["relative_path"]
    sample = db.session.query(Dsample).options(load_only(*fields)).filter(dataset_id==dataset_id and \
                segment_id==segment_id == clip_id==clip_id).first()
    res = {}
    if sample:
        res["clip_path"] = os.path.join(BaseConfig.DATASET_ROOT_PATH, sample)

    return str(res).replace("\'", "\"")
    

if __name__ == '__main__':
    app.run(debug=True)
