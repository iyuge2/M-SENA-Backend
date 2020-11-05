import os
import json
import xlwt
import subprocess
import pandas as pd

from tqdm import tqdm

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

app.config.from_object(os.environ['APP_SETTINGS'])

db = SQLAlchemy(app)

# file = UploadSet('file', VIDEOS+AUDIOS+TEXT)
# configure_uploads(app, file)

from database import *

@app.route('/')
def hello_world():
    return "Welcome to M-SENA Platform!"

@app.route('/login', methods=['POST'])
def login():
    return "success"

"""
Data-End
"""

@app.route('/data/add_dataset', methods=['POST'])
def add_dataset():
    if "path" not in request.form:
        return "Missing parameter(s)"

    data_config_path = os.path.join(request.form['path'], 'config.json')
    if not os.path.exists(data_config_path):
        return "Not found config.json in the " + request.form['path'] + " directory!"

    with open(data_config_path, 'r') as f:
        dataset_config = json.loads(f.read())

    # check model name
    names = db.session.query(Dataset).filter_by(name=dataset_config['name']).first()
    if names:
        return  dataset_config['name'] + " has existed!"

    try:
        # check data format
        for d_type in ['text_format', 'audio_format', 'video_format']:
            if dataset_config[d_type] not in app.config[d_type.upper()]:
                return "Error format in " + d_type + '. Only ' + \
                    '/'.join(app.config[d_type.upper()]) + ' support'
        # add new dataset to database
        payload = Dataset(
            name=dataset_config['name'],
            path=request.form['path'],
            audio_dir=dataset_config['audio_dir'],
            faces_dir=dataset_config['faces_dir'],
            label_path=dataset_config['label_path'],
            language=dataset_config['language'],
            label_type=dataset_config['label_type'],
            text_format=dataset_config['text_format'],
            audio_format=dataset_config['audio_format'],
            video_format=dataset_config['video_format'],
            has_feature=len(dataset_config['features']) > 0,
            is_locked=strBool(dataset_config['is_locked']),
            description=dataset_config['description'] if 'description' in dataset_config else ""
        )
        db.session.add(payload)

        # scan dataset for sample table
        label_path, raw_path = dataset_config['label_path'], dataset_config['raw_data_dir']

        label_df = pd.read_csv(os.path.join(request.form['path'], label_path), 
                                dtype={"video_id": "str", "clip_id": "str", "text": "str"})

        for i in tqdm(range(len(label_df))):
            video_id, clip_id, text, label, annotation, mode = \
                label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation', 'mode']]

            m_by = 0 if label else -1

            cur_video_path = os.path.join(request.form['path'], raw_path, str(video_id), \
                            str(clip_id)+"." + dataset_config['video_format'])
            # print(video_id, clip_id, text, label, annotation, mode)

            payload2 = Dsample(
                dataset_name=dataset_config['name'],
                video_id=str(video_id),
                clip_id=str(clip_id),
                video_path=cur_video_path,
                text=text,
                sample_mode=mode,
                label_value=label,
                label_by=m_by
            )
            db.session.add(payload2)

        db.session.commit()
    except Exception as e:
        # print(e)
        return "failed."
    return "success"

@app.route('/data/<dataset_name>/delete', methods=['POST'])
def delete_dataset(dataset_name):
    try:
        cur_dataset = db.session.query(Dataset).get(dataset_name)
        print(cur_dataset)
        # delete samples
        samples = db.session.query(Dsample).filter_by(dataset_name=cur_dataset.name).all()
        print(len(samples))
        for sample in samples:
            db.session.delete(sample)
        db.session.delete(cur_dataset)
        
        db.session.commit()
    except Exception as e:
        print(e)
        return "failed"
    
    return "success"


@app.route('/data/info', methods=['GET'])
def get_datasets_info():
    datasets = db.session.query(Dataset).all()
    res = {}
    for dataset in datasets:
        p = dataset.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res[p['name']] = p
    return str(res).replace("\'", "\"")

@app.route('/data/<dataset_name>/details', methods=['GET'])
def get_dataset_details(dataset_name):
    datasets = db.session.query(Dataset).get(dataset_name)
    res = {}
    if datasets:
        p = datasets.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res["base_info"] = p
        # details
        samples = db.session.query(Dsample).filter_by(dataset_name=dataset_name).all()
        sample_res = {}
        for sample in samples:
            p = sample.__dict__.copy()
            p.pop('_sa_instance_state', None)
            sample_res[str(p['video_id']) + "_" + str(p['clip_id'])] = p
        res["detail_info"] = sample_res
    else:
        return dataset_name + " is not existed!"

    return str(res).replace("\'", "\"")

@app.route('/data/<sample_id>', methods=['GET'])
def get_clip_video(sample_id):
    sample = db.session.query(Dsample).get(sample_id)
    res = {}
    if sample:
        res["clip_path"] = sample.video_path

    return str(res).replace("\'", "\"")

"""
Model-End
"""

"""
Test-End
"""

"""
Tasks
"""

if __name__ == '__main__':
    app.run()
