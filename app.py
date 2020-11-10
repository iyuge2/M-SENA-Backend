import os
import json
import xlwt
import subprocess
import pandas as pd

from tqdm import tqdm

from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import load_only
from flask_cors import CORS, cross_origin
from flask_uploads import UploadSet, configure_uploads
from flask import make_response

from config import BaseConfig
from functions import strBool

# def after_request(response):
#     response.headers['Access-Control-Allow-Origin'] = "*"
#     response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,Referer,User-Agent'
#     response.headers['Access-Control-Allow-Credentials'] = 'true'
#     return response

# session.clear()
app = Flask(__name__)
# support cross-domain access
CORS(app)

app.config.from_object(os.environ['APP_SETTINGS'])

db = SQLAlchemy(app)

# file = UploadSet('file', VIDEOS+AUDIOS+TEXT)
# configure_uploads(app, file)

@app.after_request
def after(resp):
    '''
    被after_request钩子函数装饰过的视图函数 
    ，会在请求得到响应后返回给用户前调用，也就是说，这个时候，
    请求已经被app.route装饰的函数响应过了，已经形成了response，这个时
    候我们可以对response进行一些列操作，我们在这个钩子函数中添加headers，所有的url跨域请求都会允许！！！
    '''
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp

from database import *

@app.route('/')
def hello_world():
    return "Welcome to M-SENA Platform!"

@app.route('/test', methods=['GET'])
def login():
    ret = {
        "code": 200,
        "msg": 'success',
        "res": "hello world",
    }
    return after(ret)

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

    # check dataset name
    dataset_name = dataset_config['name']
    names = db.session.query(Dataset).filter_by(name=dataset_name).first()
    if names:
        return  dataset_name + " has existed!"

    try:
        # check data format
        for d_type in ['text_format', 'audio_format', 'video_format']:
            if dataset_config[d_type] not in app.config[d_type.upper()]:
                return "Error format in " + d_type + '. Only ' + \
                    '/'.join(app.config[d_type.upper()]) + ' support'
        has_feature = len(dataset_config['features']) > 0
        # add new dataset to database
        payload = Dataset(
            name=dataset_name,
            path=request.form['path'],
            audio_dir=dataset_config['audio_dir'],
            faces_dir=dataset_config['faces_dir'],
            label_path=dataset_config['label_path'],
            language=dataset_config['language'],
            label_type=dataset_config['label_type'],
            text_format=dataset_config['text_format'],
            audio_format=dataset_config['audio_format'],
            video_format=dataset_config['video_format'],
            has_feature=has_feature,
            is_locked=strBool(dataset_config['is_locked']),
            description=dataset_config['description'] if 'description' in dataset_config else ""
        )
        db.session.add(payload)

        # add features
        if has_feature:
            for k, v in dataset_config['features'].items():
                payload = Dfeature(
                    dataset_name=dataset_name,
                    feature_path = v['path'],
                    input_lens=v['input_lens'],
                    feature_dims=v['feature_dims'],
                    description=v['description']
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

            payload = Dsample(
                dataset_name=dataset_config['name'],
                video_id=str(video_id),
                clip_id=str(clip_id),
                video_path=cur_video_path,
                text=text,
                sample_mode=mode,
                label_value=label,
                label_by=m_by
            )
            db.session.add(payload)

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
