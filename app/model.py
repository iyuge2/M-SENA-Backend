import json
import logging
import os
import subprocess

from constants import *
from database import Result, Task
from flask import request

from app import app, db
from app.user import check_token


logger = logging.getLogger('app')


@app.route('/modelEnd/modelList', methods=['GET'])
def get_model_list():
    logger.debug("API called: /modelEnd/modelList")
    try:
        with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
            config = json.load(fp)["MODELS"]
        res = []
        for name, items in config.items():
            p = {}
            p['model_name'] = name
            for k, v in items.items():
                p[k] = v
            res.append(p)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "modelList": res}


@app.route('/modelEnd/startTraining', methods=['POST'])
def train_model():
    logger.debug("API called: /modelEnd/startTraining")
    try:
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
            '--description', data['description'],
            '--feature_T', data['featureT'],
            '--feature_A', data['featureA'],
            '--feature_V', data['featureV'],
        ]
        p = subprocess.Popen(cmd_page, close_fds=True)

        payload.task_pid = p.pid
        db.session.commit()
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/modelEnd/getArgs', methods=['POST'])
def get_args():
    logger.debug("API called: /modelEnd/getArgs")
    try:
        requests = json.loads(request.get_data())
        model_name = requests['model']
        dataset_name = requests['dataset']
        with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as fp:
            config = json.load(fp)["MODELS"]
        args = config[model_name]['args'][dataset_name]
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "args": json.dumps(args)}
