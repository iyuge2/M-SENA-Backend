import json
import logging
import os
from multiprocessing import Process
from pathlib import Path

from config.constants import *
from database import Result, Task
from flask import request
from MMSA import SENA_run, get_config_regression, get_config_tune, get_citations

from app import app, db, progress_queue
from app.user import check_token

logger = logging.getLogger('app')


@app.route('/modelEnd/getCitations', methods=['GET'])
def get_paper_citations():
    logger.debug("API called: /modelEnd/getCitations")
    try:
        res = []
        cites = get_citations()['models']
        for k, v in cites.items():
            tmp = {'modelName':k}
            tmp.update(v)
            res.append(tmp)
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
            task_type=2 if data['isTune'] else 1,
            task_pid=10000,
            state=0
        )

        db.session.add(payload)
        db.session.flush()
        task_id = payload.task_id

        p = Process(target=SENA_run, kwargs={
            'task_id': task_id,
            'progress_q': progress_queue,
            'db_url': DATABASE_URL,
            # use default parameters if not in advanced mode
            'parameters': data['args'] if data['advanced'] else '',
            'model_name': data['model'],
            'dataset_name': data['dataset'],
            'is_tune': data['isTune'],
            'tune_times': data['tuneTimes'],
            'feature_T': data['featureT'],
            'feature_A': data['featureA'],
            'feature_V': data['featureV'],
            'model_save_dir': MODEL_SAVE_PATH,
            'res_save_dir': RES_SAVE_PATH,
            'log_dir': Path(LOG_FILE_PATH).parent,
            'gpu_ids': [], # auto detect gpu
            'num_workers': 0, # set to 0 to avoid multiprocessing errors
            'seed': data['seed'],
            'desc': data['description']
        })
        p.start()
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
        is_tune = requests['isTune']
        if is_tune:
            args = get_config_tune(model_name=model_name, dataset_name=dataset_name, random_choice=False)
        else:
            args = get_config_regression(model_name=model_name, dataset_name=dataset_name)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "args": json.dumps(args)}
