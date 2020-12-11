import json
import os
import subprocess
from collections import Counter

import pandas as pd
import xlwt
from flask import Flask, make_response, request, send_from_directory
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads
from sqlalchemy.orm import load_only
from tqdm import tqdm

from config import BaseConfig
from constants import *
from functions import strBool
from httpServer import run_http_server

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

"""
TEST
"""

# @app.route('/')
# def hello_world():
#     return "Welcome to M-SENA Platform!"

# @app.route('/test', methods=['GET'])
# def login():
#     print('test')
#     ret = {
#         "code": SUCCESS_CODE,
#         "msg": 'success',
#         "res": "hello world",
#     }
#     return ret

"""
Data-End
"""
@app.route('/dataEnd/addDataset', methods=['POST'])
def add_dataset():
    data = json.loads(request.get_data())
    print(data)
    path = data['datasetPath']
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

        # scan dataset for sample table
        label_path, raw_path = dataset_config['label_path'], dataset_config['raw_video_dir']

        label_df = pd.read_csv(os.path.join(path, label_path), 
                                dtype={"video_id": "str", "clip_id": "str", "text": "str"})

        for i in tqdm(range(len(label_df))):
            video_id, clip_id, text, label, annotation, mode = \
                label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation', 'mode']]
            
            m_by = 0 if label != -100 else -1

            cur_video_path = os.path.join(dataset_config['path'], raw_path, str(video_id), \
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
                annotation=annotation,
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
    dataset_name = json.loads(request.get_data())['datasetName']
    try:
        cur_dataset = db.session.query(Dataset).get(dataset_name)
        # delete samples
        db.session.query(Dsample).filter_by(dataset_name=cur_dataset.dataset_name).delete()
        db.session.delete(cur_dataset)
        db.session.commit()
    except Exception as e:
        return {"code": ERROR_CODE, "msg": str(e)}
    
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/getDatasetList', methods=['POST'])
def get_datasets_info():
    data = json.loads(request.get_data())
    page,pageSize = data['pageNo'], data['pageSize']

    # print(page, pageSize)
    datasets = db.session.query(Dataset)

    if data['unlocked']:
        datasets = datasets.filter_by(is_locked=False).all()
    else:
        datasets = datasets.all()

    print(datasets)
    res = []
    for dataset in datasets:
        samples = db.session.query(Dsample.dataset_name).filter_by(dataset_name=dataset.dataset_name).all()
        p = dataset.__dict__.copy()
        p.pop('_sa_instance_state', None)
        p['capacity'] = len(samples)
        res.append(p)

    totolCount = len(res)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    res = res[start_i:end_i] 

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "datasetList": res}

@app.route('/dataEnd/getMetaData', methods=['POST'])
def get_meta_data():
    dataset_name = json.loads(request.get_data())['datasetName']

    datasets = db.session.query(Dataset).get(dataset_name)
    res = datasets.__dict__.copy()
    res.pop('_sa_instance_state', None)

    samples = db.session.query(Dsample).filter_by(dataset_name=dataset_name)
    res['unlabelled'] = len(samples.filter_by(label_by=-1).all())
    res['human'] = len(samples.filter_by(label_by=0).all())
    res['easy'] = len(samples.filter_by(label_by=1).all())
    res['medium'] = len(samples.filter_by(label_by=2).all())
    res['hard'] = len(samples.filter_by(label_by=3).all())
    res['totalCount'] = len(samples.all())

    annotations = [sample.annotation for sample in samples]
    res['classCount'] = Counter(annotations)
    
    return {"code": SUCCESS_CODE, "msg": "success", "data": res}

@app.route('/dataEnd/getDetails', methods=['POST'])
def get_dataset_details():
    data = json.loads(request.get_data())
    dataset_name = data['datasetName']
    page,pageSize = data['pageNo'], data['pageSize']
    print(data)

    samples = db.session.query(Dsample).filter_by(dataset_name=data['datasetName'])
    if data['difficulty'] != 'All':
        samples = samples.filter_by(model_name=data['model_name'])
    if data['sentiment_filter'] != 'All':
        samples = samples.filter_by(annotation=LABEL_BY_N2I[data['sentiment']])
    if data['data_mode_filter'] != 'All':
        samples = samples.filter_by(data_mode=data['data_mode'])

    samples = samples.all()

    ret = []
    for sample in samples:
        p = sample.__dict__.copy()
        p.pop('_sa_instance_state', None)
        if p['label_by'] == 0:
            p['manualLabel'] = p['label_value']
        else:
            p['manualLabel'] = '-'
        # p['video_path'] = os.path.join(DATASET_SERVER_IP, p['video_path'])
        ret.append(p)

    totolCount = len(ret)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    ret = ret[start_i:end_i]

    return {"code": SUCCESS_CODE, "msg": "success", "data": ret, "totalCount": totolCount}

@app.route('/dataEnd/getVideoInfoByID', methods=['POST'])
def get_clip_video():
    sample_id = json.loads(request.get_data())['sample_id']
    sample = db.session.query(Dsample).get(sample_id)
    res = {
        'video_url': os.path.join(DATASET_SERVER_IP, sample.video_path),
    }

    return {"code": SUCCESS_CODE, "msg": 'success', 'data': res}

@app.route('/dataEnd/unlockDataset', methods=["POST"])
def unlock_dataset():
    print('test')
    dataset_name = json.loads(request.get_data())['dataset_name']
    dataset = db.session.query(Dataset).get(dataset_name)
    dataset.is_locked = False
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/lockDataset', methods=["POST"])
def lock_dataset():
    dataset_name = json.loads(request.get_data())['dataset_name']
    dataset = db.session.query(Dataset).get(dataset_name)
    dataset.is_locked = True
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

"""
DATA-Labeling
"""
@app.route('/dataEnd/submitLabelResult', methods=["POST"])
def update_label():
    results = json.loads(request.get_data())['resultList']
    for res in results:
        sample = db.session.query(Dsample).get(res['sampleId'])
        sample.label_value = LABEL_NAME_N2I[res['result']]
        sample.annotation = res['result']
    db.session.commit()
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/getNextSampleId', methods=["POST"])
def get_unlabeled_sample():
    datasetName = json.loads(request.get_data())['datasetName']
    sample = db.session.query(Dsample).filter_by(dataset_name=datasetName).first()
    if sample:
        return {"code": SUCCESS_CODE, "msg": 'success', "nextVideoSampleId": sample.sample_id}
    else:
        return {"code": WARNING_CODE, "msg": "Completed!"}

@app.route('/dataEnd/getVideoLabelInfoById', methods=["POST"])
def get_video_url():
    sampleId = json.loads(request.get_data())['sampleId']
    sample = db.session.query(Dsample).get(sampleId)
    return {"code": SUCCESS_CODE, "msg": 'success', "text": sample.text, "videoUrl": sample.video_path}

@app.route('/dataEnd/startActiveLearning', methods=["POST"])
def run_activeLearning():
    data = json.loads(request.get_data())

    payload = Task(
        dataset_name=data['datasetName'],
        model_name=data['classifier'],
        task_type=0,
        task_pid=10000,
        state=0
    )

    db.session.add(payload)
    db.session.flush()
    task_id = payload.task_id
    db.session.commit()

    cmd_page = [
        'python', os.path.join(AL_CODES_PATH, 'run.py'),
        '--use_db', True,
        '--classifier', data['classifier'],
        '--selector', data['selector'],
        '--datasetName', data['datasetName'],
        '--classifier_parameters', data['classifier_parameters'],
        '--selector_parameters', data['selector_parameters'],
        '--task_id', str(task_id)
    ]
    p = subprocess.Popen(cmd_page, close_fds=True)

    payload.task_pid = p.pid
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}
    

@app.route('/dataEnd/getALModels', methods=["GET"])
def get_classifier():
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)
        classfiers = list(config["classifiers"].keys())
        selectors = list(config["selectors"].keys())
    return {"code": SUCCESS_CODE, "msg": "success", "classifierList": classfiers, "selectorList": selectors}

@app.route('/dataEnd/getClassifierConfig', methods=["POST"])
def get_classifier_config():
    classifier_name = json.loads(request.get_data())['classifier']
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["classifiers"][classifier_name]
    return {"code": SUCCESS_CODE, "msg": "success", "args": json.dumps(config['args'])}

@app.route('/dataEnd/getSelectorConfig', methods=["POST"])
def get_selector_config():
    selector_name = json.loads(request.get_data())['selector']
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["selectors"][selector_name]
    return {"code": SUCCESS_CODE, "msg": "success", "args": json.dumps(config['args'])}

"""
Model-End
"""
@app.route('/modelEnd/modelList', methods=['GET'])
def get_model_list():
    with open(os.path.join(CODE_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["MODELS"]
    res = []
    for name, items in config.items():
        p = {}
        p['model_name'] = name
        for k, v in items.items():
            p[k] = v
        res.append(p)

    return {"code": SUCCESS_CODE, "msg": 'success', "modelList": res}

@app.route('/modelEnd/startTraining', methods=['POST'])
def train_model():
    data = json.loads(request.get_data())

    payload = Task(
        dataset_name=data['dataset'],
        model_name=data['model'],
        task_type=2,
        task_pid=10000,
        state=0
    )

    db.session.add(payload)
    db.session.flush()
    task_id = payload.task_id
    db.session.commit()

    cmd_page = [
        'python', os.path.join(CODE_PATH, 'run.py'),
        '--run_mode', data['mode'],
        '--modelName', data['model'],
        '--datasetName', data['dataset'],
        '--parameters', data['args'],
        '--description', data['description'],
        '--task_id', str(task_id)
        # '--use_stored', data['useStored']
    ]
    p = subprocess.Popen(cmd_page, close_fds=True)

    payload.task_pid = p.pid
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/settings/getAllSettings', methods=['GET'])
def get_settings():
    with open(os.path.join(CODE_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["MODELS"]
        model_names = list(config.keys())

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
    print(data)

    results = db.session.query(Result)
    if data['model_name'] != 'All':
        results = results.filter_by(model_name=data['model_name'])
    if data['dataset_name'] != 'All':
        results = results.filter_by(dataset_name=data['dataset_name'])
    if data['is_tuning'] != 'Both':
        results = results.filter_by(is_tuning=data['is_tuning'])

    results = results.all()
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
    model_name = json.loads(request.get_data())['model']
    with open(os.path.join(CODE_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["MODELS"]
    args = config[model_name]['args']
    return {"code": SUCCESS_CODE, "msg": 'success', "args":json.dumps(args)}

@app.route('/modelEnd/setDefaultParams', methods=['POST'])
def set_default_args():
    result_id = json.loads(request.get_data())['id']
    cur_result = db.session.query(Result).get(result_id)
    # revise config.json in model codes
    with open(os.path.join(CODE_PATH, 'config.json'), 'r') as f:
        model_config = json.load(f)
    model_config[cur_result.model_name]['args'] = cur_result.args
    with open(os.path.join(CODE_PATH, 'config.json'), 'w') as f:
        json.dumps(model_config, f, indent=4)

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/modelEnd/delResult', methods=['POST'])
def del_results():
    result_id = json.loads(request.get_data())['id']
    try:
        cur_result = db.session.query(Result).get(result_id)
        db.session.delete(cur_result)
        e_results = db.session.query(EResult).filter_by(result_id=result_id).all()
        for e_result in e_results:
            db.session.delete(e_result)
        db.session.commit()
    except Exception as e:
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}

"""
Test-End
"""
@app.route('/presentatinEnd/batchResults', methods=['POST'])
def batch_test():
    data = json.loads(request.get_data())


"""
Tasks
"""
@app.route('/task/getTaskList', methods=["GET"])
def get_task_list():
    tasks = db.session.query(Task).all()
    
    task_type_dict = {
        0: 'Machine Labeling',
        1: 'Model Training',
        2: 'Model Tuning',
        3: 'Model Test'
    }
    run_tasks, error_tasks, terminate_tasks, finished_tasks = [], [], [], []
    for task in tasks:
        p = task.__dict__.copy()
        p.pop('_sa_instance_state', None)
        p['task_type'] = task_type_dict[p['task_type']]
        if p['state'] == 0:
            run_tasks.append(p)
        elif p['state'] == 1:
            finished_tasks.append(p)
        elif p['state'] == 2:
            error_tasks.append(p)
        else:
            terminate_tasks.append(p)
    
    ret = {
        'runList': run_tasks,
        'errList': error_tasks,
        'termList': terminate_tasks,
        'finList': finished_tasks,
        'totalNum': len(tasks)
    }

    return {"code": SUCCESS_CODE, "msg": 'success', "data": ret}

@app.route('/task/stopTask', methods=['POST'])
def stop_task():
    data = json.loads(request.get_data())
    task_id = data['task_id']
    cur_task = db.session.query(Task).get(task_id)

    cmd = 'kill -9 ' + str(cur_task.task_pid) 
    os.system(cmd)

    cur_task.state = 3
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/task/delTask', methods=['POST'])
def delete_task():
    data = json.loads(request.get_data())
    print(data)
    task_id = data['task_id']
    cur_task = db.session.query(Task).get(task_id)
    db.session.delete(cur_task)
    db.session.commit()
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/task/delAllTask', methods=["GET"])
def del_unrun_tasks():
    tasks = db.session.query(Task).all()
    for task in tasks:
        if task.state != 0:
            db.session.delete(task)

    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.after_request
def after(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin') or 'http://127.0.0.1:1024'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type' 
    # resp.headers['Access-Control-Allow-Credentials'] = 'true'
    return resp

if __name__ == "app":
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)

if __name__ == "__main__":
    app.run()
