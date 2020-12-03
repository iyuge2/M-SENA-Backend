import os
import json
import xlwt
import subprocess
import pandas as pd
import subprocess

from tqdm import tqdm

from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import load_only
from flask_cors import CORS, cross_origin
from flask_uploads import UploadSet, configure_uploads
from flask import make_response

from config import BaseConfig
from functions import strBool
from constants import *

# session.clear()
app = Flask(__name__)

# app.after_request(after_request)
# support cross-domain access
# CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:1024"}} )

app.config.from_object(os.environ['APP_SETTINGS'])

db = SQLAlchemy(app)

# file = UploadSet('file', VIDEOS+AUDIOS+TEXT)
# configure_uploads(app, file)

from database import *

@app.route('/')
def hello_world():
    return "Welcome to M-SENA Platform!"

@app.route('/test', methods=['GET'])
def login():
    print('test')
    ret = {
        "code": SUCCESS_CODE,
        "msg": 'success',
        "res": "hello world",
    }
    return ret

"""
Data-End
"""
@app.route('/dataEnd/addDataset', methods=['POST'])
def add_dataset():
    path = json.loads(request.get_data())['path']
    data_config_path = os.path.join(path, 'config.json')
    if not os.path.exists(data_config_path):
        return {"code": ERROR_CODE, "msg": data_config_path + " is not exist!"}

    with open(data_config_path, 'r') as f:
        dataset_config = json.loads(f.read())

    # check dataset name
    dataset_name = dataset_config['name']
    names = db.session.query(Dataset).filter_by(dataset_name=dataset_name).first()
    if names:
        return {"code": WARNING_CODE, "msg": dataset_name + " has existed!"}
    try:
        # check data format
        for d_type in ['text_format', 'audio_format', 'video_format']:
            if dataset_config[d_type] not in SUPPORT_FORMAT[d_type]:
                return {"code": ERROR_CODE, "msg": "Error format in " + d_type + '. Only ' + \
                            '/'.join(SUPPORT_FORMAT[d_type]) + ' support'}

        has_feature = len(dataset_config['features']) > 0
        # add new dataset to database
        payload = Dataset(
            dataset_name=dataset_name,
            dataset_path=path,
            language=dataset_config['language'],
            label_path=dataset_config['label_path'],
            text_format=dataset_config['text_format'],
            audio_format=dataset_config['audio_format'],
            video_format=dataset_config['video_format'],
            raw_video_dir=dataset_config['raw_video_dir'],
            audio_dir=dataset_config['audio_dir'],
            faces_dir=dataset_config['faces_dir'],
            has_feature=has_feature,
            is_locked=dataset_config['is_locked'],
            description=dataset_config['description'] if 'description' in dataset_config else ""
        )
        db.session.add(payload)

        # add features
        if has_feature:
            for k, v in dataset_config['features'].items():
                payload = Dfeature(
                    dataset_name=dataset_name,
                    feature_path = os.path.join(path, v['feature_path']),
                    seq_lens=json.dumps(v['seq_lens']),
                    feature_dims=json.dumps(v['feature_dims']),
                    description=v['description']
                )
                db.session.add(payload)

        # scan dataset for sample table
        label_path, raw_path = dataset_config['label_path'], dataset_config['raw_video_dir']

        label_df = pd.read_csv(os.path.join(path, label_path), 
                                dtype={"video_id": "str", "clip_id": "str", "text": "str"})

        for i in tqdm(range(len(label_df))):
            video_id, clip_id, text, label, annotation, mode = \
                label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation', 'mode']]
            
            m_by = 0 if label else -1

            cur_video_path = os.path.join(path, raw_path, str(video_id), \
                            str(clip_id)+"." + dataset_config['video_format'])
            # print(video_id, clip_id, text, label, annotation, mode)

            payload = Dsample(
                dataset_name=dataset_name,
                video_id=video_id,
                clip_id=clip_id,
                video_path=cur_video_path,
                text=text,
                data_mode=mode,
                label_value=label,
                label_by=m_by
            )
            db.session.add(payload)

        db.session.commit()
    except Exception as e:
        # print(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success"}

@app.route('/dataEnd/deleteDataset', methods=['POST'])
def delete_dataset():
    dataset_name = json.loads(request.get_data())['dataset_name']
    try:
        cur_dataset = db.session.query(Dataset).get(dataset_name)
        # delete samples
        samples = db.session.query(Dsample).filter_by(dataset_name=cur_dataset.dataset_name).all()
        for sample in samples:
            db.session.delete(sample)
        features = db.session.query(Dfeature).filter_by(dataset_name=cur_dataset.dataset_name).all()
        for feature in features:
            db.session.delete(feature)

        db.session.delete(cur_dataset)
        db.session.commit()
    except Exception as e:
        print(e)
        return {"code": ERROR_CODE, "msg": 'failed', "res": ""}
    
    return {"code": SUCCESS_CODE, "msg": 'success', "res": ""}


@app.route('/dataEnd/info', methods=['POST'])
def get_datasets_info():
    page = request.form['pageNo']
    pageSize = request.form['pageSize']
    
    # print(page, pageSize)
    datasets = db.session.query(Dataset).all()
    res = []
    for dataset in datasets:
        p = dataset.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res.append(p)
    totolCount = len(res)

    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}

    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    res = res[start_i:end_i] 

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "res": res}

@app.route('/dataEnd/<dataset_name>/details', methods=['GET'])
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

    return {"code": SUCCESS_CODE, "msg": res}

@app.route('/dataEnd/<sample_id>', methods=['GET'])
def get_clip_video(sample_id):
    sample = db.session.query(Dsample).get(sample_id)
    res = {}
    if sample:
        res["video_path"] = sample.video_path

    return {"code": SUCCESS_CODE, "msg": res}

"""
Model-End
"""
@app.route('/modelEnd/scanModel', methods=['GET'])
def add_model():
    with open(os.path.join(CODE_PATH, 'config.json'), 'r') as f:
        model_config = json.load(f)

    db.session.query(Model).delete()
    try:
        for k,v in model_config.items():
            payload = Model(
                model_name=k,
                args=json.dumps(v['args']),
                paper_name=v['paper_name'],
                paper_url=v['paper_url'],
                description=v['description']
            )
            db.session.add(payload)
        db.session.commit()
    except Exception as e:
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/modelEnd/modelList', methods=['POST'])
def get_model_list():
    data = json.loads(request.get_data())
    page,pageSize = data['pageNo'], data['pageSize']
    # print(page, pageSize)
    models = db.session.query(Model)
    res = []
    for model in models:
        p = model.__dict__.copy()
        p.pop('_sa_instance_state', None)
        res.append(p)

    totolCount = len(res)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    res = res[start_i:end_i] 

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "modelList": res}

@app.route('/modelEnd/delModel', methods=['POST'])
def del_model():
    data = json.loads(request.get_data())
    try:
        model = db.session.query(Model).get(data['model'])
        db.session.delete(model)
        db.session.commit()
    except Exception as e:
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/modelEnd/startTraining', methods=['POST'])
def train_model():
    data = json.loads(request.get_data())
    print(data)
    cur_model = db.session.query(Model).get(data['model'])
    # cur_dataset = db.session.query(Dataset).get(data['dataset_name'])
    cmd_page = [
        'python', os.path.join(CODE_PATH, 'run.py'),
        '--debug_mode', data['mode'].lower(),
        '--modelName', data['model'],
        '--datasetName', data['dataset'],
        '--parameters', cur_model.args
    ]
    p = subprocess.Popen(cmd_page, close_fds=True)
    print(p.pid)
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/settings/getAllSettings', methods=['GET'])
def get_settings():
    model_names = db.session.query(Model.model_name).all()
    model_names = [v[0] for v in model_names]
    datasets = db.session.query(Dataset).all()
    ret_datas = [] 
    for dataset in datasets:
        cur_data = {}
        cur_data["name"] = dataset.dataset_name
        label_path = os.path.join(dataset.dataset_path, dataset.label_path)
        df = pd.read_csv(label_path)
        cur_data['sentiments'] = list(set(df['annotation'].values))
        ret_datas.append(cur_data)
    return {"code": SUCCESS_CODE, "msg": 'success', "models": model_names, \
            "datasets": ret_datas}

@app.route('/modelEnd/getResults', methods=["POST"])
def get_results():
    data = json.loads(request.get_data())

    # sql_exec = 'SELECT * FROM Result WHERE'
    # if data['model_name'] != 'All':
    #     sql_exec = sql_exec + " model_name=" + data['model'] + ' AND'
    # if data['dataset_name'] != 'All':
    #     sql_exec = sql_exec + " dataset_name=" + data['dataset'] + ' AND'
    # if data['is_tuning'] == 'Tune':
    #     sql_exec = sql_exec + " is_tuning=True AND"
    # elif data['is_tuning'] == 'Normal':
    #     sql_exec = sql_exec + " is_tuning=False AND"

    # if sql_exec[-5:] == 'WHERE':
    #     sql_exec = sql_exec[:-5]
    # elif sql_exec[-3:] == 'AND':
    #     sql_exec = sql_exec[:-3]
    # results = db.session.execute(sql_exec).fetchall()
    results = db.session.query(Result).all()
    if data['model_name'] != 'All':
        results = results.filter_by(model_name=data['model_name'])
    if data['dataset_name'] != 'All':
        results = results.filter_by(dataset_name=data['dataset_name'])
    if data['is_tuning'] == 'Tune':
        results = results.filter_by(is_tuning=True)
    elif data['is_tuning'] == 'Normal':
        results = results.filter_by(is_tuning=False)

    ret = []
    print(results)
    for result in results:
        p = result.__dict__.copy()
        p.pop('_sa_instance_state', None)
        cur_id = p['result_id']
        e_result = db.session.query(EResult).filter_by(result_id=cur_id,epoch_num=-1).first()
        e_result = e_result.__dict__.copy()
        for k in ['loss_value', 'accuracy', 'f1', 'mae', 'corr']:
            p[k] = e_result[k]
        ret.append(p)
    
    print(ret)

    page,pageSize = data['pageNo'], data['pageSize']
    
    totolCount = len(ret)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    ret = ret[start_i:end_i]

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "results": ret}

@app.route('/modelEnd/getArgs', methods=['POST'])
def get_args():
    data = json.loads(request.get_data())
    model = db.session.query(Model).get(data['model'])
    # args = json.dumps(json.loads(model.args), indent=4)
    args = model.args
    print(args)
    return {"code": SUCCESS_CODE, "msg": 'success', "args":args}

@app.route('/modelEnd/delResult', methods=['POST'])
def del_results():
    result_id = json.loads(request.get_data())


"""
Test-End
"""

"""
Tasks
"""

@app.after_request
def after(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin') or 'http://127.0.0.1:1024'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type' 
    # resp.headers['Access-Control-Allow-Credentials'] = 'true'
    return resp

if __name__ == '__main__':
    app.run()
