import json
import os
import subprocess
from glob import glob
from collections import Counter
from datetime import datetime

import pandas as pd
import xlwt
from flask import Flask, make_response, request, send_from_directory
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads
from sqlalchemy.orm import load_only
from sqlalchemy import or_, and_, not_, desc, asc
from tqdm import tqdm

from constants import *
from database import *
from httpServer import run_http_server

# session.clear()
app = Flask(__name__)

# app.after_request(after_request)
# support cross-domain access
# CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:1024"}} )

app.config.from_object(os.environ['APP_SETTINGS'])

db = SQLAlchemy(app)

"""
TEST
"""

"""
Data-End
"""
@app.route('/dataEnd/scanDatasets', methods=['GET'])
def scan_datasets():
    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
        dataset_config = json.load(fp)
    
    for dataset_name, configs in dataset_config.items():
        db.session.query(Dsample).filter_by(dataset_name=dataset_name).delete()
        # scan dataset for sample table
        label_df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, configs['label_path']), 
                                dtype={"video_id": "str", "clip_id": "str", "text": "str"})

        for i in tqdm(range(len(label_df))):
            video_id, clip_id, text, label, annotation, mode, label_by = \
                label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation', 'mode', 'label_by']]
            
            cur_video_path = os.path.join(configs['raw_video_dir'], video_id, \
                                            clip_id+"." + configs['video_format'])
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
                label_by=label_by
            )
            db.session.add(payload)

        db.session.commit()
    return {"code": SUCCESS_CODE, "msg": "success"}

@app.route('/dataEnd/getDatasetList', methods=['POST'])
def get_datasets_info():
    data = json.loads(request.get_data())
    page,pageSize = data['pageNo'], data['pageSize']
    
    res = []
    sample = db.session.query(Dsample).first()
    if sample:
        # print(page, pageSize)
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)

        
        for k, dataset in dataset_config.items():
            p = {}
            print(dataset)
            if data['unlocked'] == False or \
                (data['unlocked'] and dataset['is_locked'] == False):
                p['datasetName'] = k
                p['status'] = 'locked' if dataset['is_locked'] else 'unlocked'
                p['language'] = dataset['language']
                p['description'] = dataset['description']
                samples = db.session.query(Dsample.dataset_name).filter_by(dataset_name=k).all()
                p['capacity'] = len(samples)
                res.append(p)

        totolCount = len(res)
        start_i = (page - 1) * pageSize
        if start_i > totolCount:
            return {"code": ERROR_CODE, "msg": 'page error!'}
        end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
        res = res[start_i:end_i]
    else:
        totolCount = 0

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "datasetList": res}

@app.route('/dataEnd/getMetaData', methods=['POST'])
def get_meta_data():
    dataset_name = json.loads(request.get_data())['datasetName']

    # print(page, pageSize)
    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
        dataset = json.load(fp)[dataset_name]

    res = {} 
    res['datasetName'] = dataset_name
    res['status'] = 'locked' if dataset['is_locked'] else 'unlocked'
    res['language'] = dataset['language']
    res['description'] = dataset['description']

    samples = db.session.query(Dsample).filter_by(dataset_name=dataset_name).all()
    
    res['totalCount'] = len(samples)

    label_bys = [sample.label_by for sample in samples]
    label_bys = Counter(label_bys)
    tmp = {}
    for k,v in label_bys.items():
        tmp[LABEL_BY_I2N[k]] = v
    res['difficultyCount'] = tmp

    annotations = [sample.annotation for sample in samples]
    res['classCount'] = Counter(annotations)

    modes = [sample.data_mode for sample in samples]
    res['typeCount'] = Counter(modes)
    
    return {"code": SUCCESS_CODE, "msg": "success", "data": res}
    
@app.route('/dataEnd/getDetails', methods=['POST'])
def get_dataset_details():
    data = json.loads(request.get_data())
    dataset_name = data['datasetName']
    page,pageSize = data['pageNo'], data['pageSize']

    samples = db.session.query(Dsample).filter_by(dataset_name=data['datasetName'])
    if data['difficulty_filter'] != 'All':
        samples = samples.filter_by(label_by=LABEL_BY_N2I[data['difficulty_filter']])
    if data['sentiment_filter'] != 'All':
        samples = samples.filter_by(annotation=data['sentiment_filter'])
    if data['data_mode_filter'] != 'All':
        samples = samples.filter_by(data_mode=data['data_mode_filter'])

    samples = samples.all()

    ret = []
    for sample in samples:
        p = sample.__dict__.copy()
        p.pop('_sa_instance_state', None)
        p['manualLabel'] = LABEL_NAME_I2N[p['label_value']] if p['label_by'] == 0 else '-'
        p['prediction'] = LABEL_NAME_I2N[p['label_value']] if p['label_by'] != 0 else '-'
        p['video_url'] = os.path.join(DATASET_SERVER_IP, p['video_path'])
        ret.append(p)

    totolCount = len(ret)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    ret = ret[start_i:end_i]

    return {"code": SUCCESS_CODE, "msg": "success", "data": ret, "totalCount": totolCount}

@app.route('/dataEnd/unlockDataset', methods=["POST"])
def unlock_dataset():
    dataset_name = json.loads(request.get_data())['datasetName']
    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
        dataset_config = json.load(fp)
    dataset_config[dataset_name]['is_locked'] = False

    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'w') as fp:
        json.dump(dataset_config, fp, indent=4)

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/lockDataset', methods=["POST"])
def lock_dataset():
    dataset_name = json.loads(request.get_data())['datasetName']
    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
        dataset_config = json.load(fp)
    dataset_config[dataset_name]['is_locked'] = True

    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'w') as fp:
        json.dump(dataset_config, fp, indent=4)

    return {"code": SUCCESS_CODE, "msg": 'success'}

"""
DATA-Labeling
"""
@app.route('/dataEnd/submitLabelResult', methods=["POST"])
def update_label():
    results = json.loads(request.get_data())['resultList']
    print(results)
    for res in results:
        if res['annotation'] in LABEL_NAME_N2I.keys():
            sample = db.session.query(Dsample).get(res['sample_id'])
            sample.label_value = LABEL_NAME_N2I[res['annotation']]
            sample.annotation = res['annotation']
            sample.label_by = 0
    db.session.commit()
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/getHardSamples', methods=["POST"])
def get_hard_samples():
    datasetName = json.loads(request.get_data())['datasetName']
    samples = db.session.query(Dsample).filter_by(dataset_name=datasetName).filter(or_(Dsample.label_by==-1, Dsample.label_by==2, Dsample.label_by==3)).all()
    if len(samples) > MANUAL_LABEL_BATCH_SIZE:
        samples = samples[:MANUAL_LABEL_BATCH_SIZE]
    print(len(samples))

    data = []
    for sample in samples:
        cur_d = {
            "sample_id": sample.sample_id,
            "text": sample.text,
            "video_url": os.path.join(DATASET_SERVER_IP, sample.video_path),
            "annotation": ''
        }
        data.append(cur_d)
    return {"code": SUCCESS_CODE, "msg": 'success', "data": data}

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
    db.session.commit()
    task_id = payload.task_id
    # db.session.commit()

    cmd_page = [
        'python', os.path.join(AL_CODES_PATH, 'run.py'),
        '--use_db', 'True',
        '--classifier', data['classifier'],
        '--selector', data['selector'],
        '--datasetName', data['datasetName'],
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

@app.route('/dataEnd/saveClassifierConfig', methods=['POST'])
def save_classifier_config():
    data = json.loads(request.get_data())
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)
    config['classifiers'][data['classifier']]['args'] = json.loads(data['args'])
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'w') as fp:
        config = json.dump(config, fp, indent=4)

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/saveSelectorConfig', methods=['POST'])
def save_selector_config():
    data = json.loads(request.get_data())
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)
    config['selectors'][data['selector']]['args'] = json.loads(data['args'])
    with open(os.path.join(AL_CODES_PATH, 'config.json'), 'w') as fp:
        config = json.dump(config, fp, indent=4)

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/dataEnd/exportDataset', methods=['POST'])
def export_dataset():
    dataset_name = json.loads(request.get_data())['datasetName']
    print(dataset_name)
    return {"code": SUCCESS_CODE, "msg": 'success'}

"""
Model-End
"""
@app.route('/modelEnd/modelList', methods=['GET'])
def get_model_list():
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
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
        task_type=1 if data['mode'] == "Train" else 2, 
        task_pid=10000,
        state=0
    )

    db.session.add(payload)
    db.session.flush()
    task_id = payload.task_id

    cmd_page = [
        'python', os.path.join(MM_CODES_PATH, 'run.py'),
        '--run_mode', data['mode'],
        '--modelName', data['model'],
        '--datasetName', data['dataset'],
        '--parameters', data['args'],
        '--tune_times', data['tuneTimes'],
        '--task_id', str(task_id),
        '--description', data['description']
    ]
    p = subprocess.Popen(cmd_page, close_fds=True)

    payload.task_pid = p.pid
    db.session.commit()

    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/settings/getAllSettings', methods=['GET'])
def get_settings():
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["MODELS"]
        model_names = list(config.keys())

    with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
        dataset_config = json.load(fp)
        dataset_names = list(config.keys())

    ret_datas = [] 
    for k, dataset in dataset_config.items():
        cur_data = {}
        cur_data['name'] = k
        cur_data['sentiments'] = [v for k, v in dataset['annotations'].items()]
        ret_datas.append(cur_data)
    
    pre_trained_models = []
    defaults = db.session.query(Result).filter_by(is_tuning='Train').all()
    for default in defaults:
        cur_name = default.model_name + '-' + default.dataset_name + '-' + str(default.result_id)
        pre_trained_models.append(cur_name)

    return {"code": SUCCESS_CODE, "msg": 'success', "models": model_names, \
            "datasets": ret_datas, "pretrained": pre_trained_models}

@app.route('/modelEnd/getResults', methods=["POST"])
def get_results():
    data = json.loads(request.get_data())

    results = db.session.query(Result)
    if data['model_name'] != 'All':
        results = results.filter_by(model_name=data['model_name'])
    if data['dataset_name'] != 'All':
        results = results.filter_by(dataset_name=data['dataset_name'])
    if data['is_tuning'] != 'Both':
        results = results.filter_by(is_tuning=data['is_tuning'])

    results = results.all()
    ret = []
    
    for result in results:
        p = result.__dict__.copy()
        p.pop('_sa_instance_state', None)
        cur_id = p['result_id']
        p['test-acc'] = result.accuracy
        p['test-f1'] = result.f1
        p['train'] = {k:[] for k in ['loss_value', 'accuracy', 'f1']}
        p['valid'] = {k:[] for k in ['loss_value', 'accuracy', 'f1']}
        p['test'] = {k:[] for k in ['loss_value', 'accuracy', 'f1']}
        e_result = db.session.query(EResult).filter_by(result_id=result.result_id).order_by(asc(EResult.epoch_num)).all() 
        e_result = e_result[1:] # remove final results
        for cur_r in e_result:
            e_res = json.loads(cur_r.results)
            for mode in ['train', 'valid', 'test']:
                for item in ['loss_value', 'accuracy', 'f1']:
                    p[mode][item].append(e_res[mode][item])
        ret.append(p)
    
    page,pageSize = data['pageNo'], data['pageSize']
    
    totolCount = len(ret)
    start_i = (page - 1) * pageSize
    if start_i > totolCount:
        return {"code": ERROR_CODE, "msg": 'page error!'}
    end_i = (start_i + pageSize) if (start_i + pageSize) <= totolCount else totolCount
    ret = ret[start_i:end_i]

    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "results": ret}

@app.route('/modelEnd/getResultDetails', methods=["POST"])
def get_results_details():
    return {"code": SUCCESS_CODE, "msg": 'success', "results": {}}

@app.route('/modelEnd/getArgs', methods=['POST'])
def get_args():
    model_name = json.loads(request.get_data())['model']
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
        config = json.load(fp)["MODELS"]
    args = config[model_name]['args']
    return {"code": SUCCESS_CODE, "msg": 'success', "args":json.dumps(args)}

@app.route('/modelEnd/setDefaultModel', methods=['POST'])
def set_default_args():
    result_id = json.loads(request.get_data())['id']
    cur_result = db.session.query(Result).get(result_id)
    # revise config.json in model codes
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as f:
        model_config = json.load(f)
    model_config['MODELS'][cur_result.model_name]['args'] = json.loads(cur_result.args)
    with open(os.path.join(MM_CODES_PATH, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

    cur_result.is_default = True
    db.session.commit()
    return {"code": SUCCESS_CODE, "msg": 'success'}

@app.route('/modelEnd/delResult', methods=['POST'])
def del_results():
    result_id = json.loads(request.get_data())['id']
    try:
        cur_result = db.session.query(Result).get(result_id)
        cur_name = cur_result.model_name + '-' + cur_result.dataset_name + '-' + str(cur_result.result_id)
        os.remove(os.path.join(MODEL_TMP_SAVE, cur_name + '.pth'))
        os.remove(os.path.join(MODEL_TMP_SAVE, cur_name + '.pkl'))
        
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
@app.route('/presentationEnd/batchResults', methods=['POST'])
def batch_test():
    data = json.loads(request.get_data())
    models = [data['primary']] + data['other']
    results = []
    for model in models:
        result_id = int(model.split('-')[-1])
        e_result = db.session.query(EResult).filter_by(result_id=result_id, epoch_num=-1).first()
        cur_result = {
            "model": model,
            "acc2": e_result.accuracy,
            "acc2Delta": 0.0,
            "f1": e_result.f1,
            "f1Delta": 0.0,
            "mae": e_result.mae,
            "maeDelta": 0.0,
            "corr": e_result.corr,
            "corrDelta": 0.0
        }
        if len(results) >= 1:
            for k in ['acc2', 'f1', 'mae', 'corr']:
                cur_result[k + 'Delta'] = results[0][k] - cur_result[k]
        results.append(cur_result)
        
    return {'code': SUCCESS_CODE, 'msg': 'success', 'result': results}

@app.route('/presentationEnd/sampleResults', methods=['POST'])
def get_sample_results():
    return {"code": SUCCESS_CODE, "msg": "success"}

@app.route('/presentationEnd/liveTranscript', methods=['POST'])
def get_live_transcript():
    res = "hello mesa"
    return {"code": SUCCESS_CODE, "msg": "success", "transcript": res}

@app.route('/presentation/liveResults', methods=['POST'])
def get_live_results():
    return {"code": SUCCESS_CODE, "msg": "success"}

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
        # p['start_time'] = p['start_time'].strftime('%Y-%m-%d %H:%M')
        # p['end_time'] = p['end_time'].strftime('%Y-%m-%d %H:%M')
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
