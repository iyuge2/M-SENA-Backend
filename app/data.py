import json
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from config.constants import *
from database import Dsample
from flask import request
from sqlalchemy.dialects.mysql import insert
from tqdm import tqdm

from app import app, db
from app.user import check_token

logger = logging.getLogger('app')


@app.route('/dataEnd/scanDatasets', methods=['POST'])
def scan_datasets():
    logger.debug("API called: /dataEnd/scanDatasets")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)

        for dataset_name, configs in dataset_config.items():
            # scan dataset for sample table
            label_df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, configs['label_path']),
                                   dtype={"video_id": "str", "clip_id": "str", "text": "str"})
            label_df = label_df.replace({np.nan: None})

            for i in tqdm(range(len(label_df))):
                video_id, clip_id, text, label, annotation, mode, label_T, label_A, label_V= \
                    label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation',
                                     'mode', 'label_T', 'label_A', 'label_V']]

                if len(text) > SQL_MAX_TEXT_LEN:
                    text = text[:SQL_MAX_TEXT_LEN-10]

                cur_video_path = os.path.join(configs['raw_video_dir'], video_id,
                                              clip_id+"." + configs['video_format'])
                # print(video_id, clip_id, text, label, annotation, mode)
                insert_stmt = insert(Dsample).values(
                    dataset_name = dataset_name,
                    video_id = video_id,
                    clip_id = clip_id,
                    video_path = cur_video_path,
                    text = text,
                    data_mode = mode,
                    label_value = label,
                    annotation = annotation,
                    label_T = label_T,
                    label_A = label_A,
                    label_V = label_V
                )
                on_duplicate_key_update_stmt = insert_stmt.on_duplicate_key_update(
                    video_path = cur_video_path,
                    text = text,
                    data_mode = mode,
                    label_value = label,
                    annotation = annotation,
                    label_T = label_T,
                    label_A = label_A,
                    label_V = label_V
                )
                db.session.execute(on_duplicate_key_update_stmt)
            db.session.commit()
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success"}


@app.route('/dataEnd/updateDataset', methods=['POST'])
def update_datasets():
    logger.debug("API called: /dataEnd/updateDataset")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = json.loads(request.get_data())['datasetName']
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)

        configs = dataset_config[dataset_name]

        # scan dataset for sample table
        label_df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, configs['label_path']),
                               dtype={"video_id": "str", "clip_id": "str", "text": "str"})
        label_df = label_df.replace({np.nan: None})

        for i in tqdm(range(len(label_df))):
            video_id, clip_id, text, label, annotation, mode, label_T, label_A, label_V = \
                label_df.loc[i, ['video_id', 'clip_id', 'text', 'label', 'annotation', 
                                 'mode', 'label_T', 'label_A', 'label_V']]

            if len(text) > SQL_MAX_TEXT_LEN:
                text = text[:SQL_MAX_TEXT_LEN-10]

            cur_video_path = os.path.join(configs['raw_video_dir'], video_id,
                                          clip_id+"." + configs['video_format'])
            # print(video_id, clip_id, text, label, annotation, mode)
            insert_stmt = insert(Dsample).values(
                dataset_name = dataset_name,
                video_id = video_id,
                clip_id = clip_id,
                video_path = cur_video_path,
                text = text,
                data_mode = mode,
                label_value = label,
                annotation = annotation,
                label_T = label_T,
                label_A = label_A,
                label_V = label_V
            )
            on_duplicate_key_update_stmt = insert_stmt.on_duplicate_key_update(
                video_path = cur_video_path,
                text = text,
                data_mode = mode,
                label_value = label,
                annotation = annotation,
                label_T = label_T,
                label_A = label_A,
                label_V = label_V
            )
            db.session.execute(on_duplicate_key_update_stmt)

        db.session.commit()
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success"}


@app.route('/dataEnd/getDatasetList', methods=['POST'])
def get_datasets_info():
    logger.debug("API called: /dataEnd/getDatasetList")
    try:
        data = json.loads(request.get_data())

        res = []
        datasets = [r.dataset_name for r in db.session.query(Dsample.dataset_name).distinct()]
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)

        for k, dataset in dataset_config.items():
            if k in datasets:
                p = {}
                if data['unlocked'] == False or \
                        (data['unlocked'] and dataset['is_locked'] == False):
                    p['datasetName'] = k
                    p['status'] = 'locked' if dataset['is_locked'] else 'unlocked'
                    p['language'] = dataset['language']
                    p['description'] = dataset['description']
                    samples = db.session.query(
                        Dsample.dataset_name).filter_by(dataset_name=k).all()
                    p['capacity'] = len(samples)
                    res.append(p)

    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "datasetList": res}


@app.route('/dataEnd/getMetaData', methods=['POST'])
def get_meta_data():
    logger.debug("API called: /dataEnd/getMetaData")
    try:
        dataset_name = json.loads(request.get_data())['datasetName']

        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset = json.load(fp)[dataset_name]

        res = {}
        res['datasetName'] = dataset_name
        res['status'] = 'locked' if dataset['is_locked'] else 'unlocked'
        res['language'] = dataset['language']
        res['description'] = dataset['description']

        samples = db.session.query(Dsample).filter_by(dataset_name=dataset_name)

        res['totalCount'] = samples.count()
        unlabeld_count = samples.filter_by(annotation=None).count()
        res['labeled'] = res['totalCount'] - unlabeld_count

        annotations = [sample.annotation for sample in samples]
        res['classCount'] = Counter(annotations)
        modes = [sample.data_mode for sample in samples]
        res['typeCount'] = Counter(modes)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "data": res}


@app.route('/dataEnd/getDetails', methods=['POST'])
def get_dataset_details():
    logger.debug("API called: /dataEnd/getDetails")
    try:
        data = json.loads(request.get_data())
        page_no, page_size = data['pageNo'], data['pageSize']
        dataset_name = data['datasetName']
        sentiment = data['sentiment_filter']
        data_mode = data['data_mode_filter']
        video_id = data['id_filter']

        samples = db.session.query(Dsample).filter(
            Dsample.dataset_name == dataset_name
        )
        if sentiment != 'All':
            samples = samples.filter(Dsample.annotation == sentiment)
        if data_mode != 'All':
            samples = samples.filter(Dsample.data_mode == data_mode)
        if video_id != '':
            samples = samples.filter(Dsample.video_id == video_id)
        totol_count = samples.count()
        samples = samples.paginate(page_no, page_size, False)

        res = []
        for sample in samples.items:
            p = sample.__dict__.copy()
            p.pop('_sa_instance_state', None)
            for k,v in p.items():
                if v == None:
                    p[k] = '-'
            p['video_url'] = os.path.join(DATASET_SERVER_IP, p['video_path'])
            res.append(p)

    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "data": res, "totalCount": totol_count}


@app.route('/dataEnd/unlockDataset', methods=["POST"])
def unlock_dataset():
    logger.debug("API called: /dataEnd/unlockDataset")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = json.loads(request.get_data())['datasetName']
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)
        dataset_config[dataset_name]['is_locked'] = False

        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'w') as fp:
            json.dump(dataset_config, fp, indent=4)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/lockDataset', methods=["POST"])
def lock_dataset():
    logger.debug("API called: /dataEnd/lockDataset")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = json.loads(request.get_data())['datasetName']
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            dataset_config = json.load(fp)
        dataset_config[dataset_name]['is_locked'] = True

        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'w') as fp:
            json.dump(dataset_config, fp, indent=4)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}
