import base64
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from glob import glob
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from Crypto.Cipher import AES
from Crypto.Util import Padding
from flask import Flask, make_response, request
from flask_sqlalchemy import SQLAlchemy
from pytz import timezone
from sqlalchemy import and_, asc, desc, not_, or_, func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql.coercions import LabeledColumnExprImpl
from tqdm import tqdm

from constants import *
from database import Annotation, Dsample, EResult, Result, Task, User, Feature
from httpServer import run_http_server
from MMSA.run_live import run_live

if MM_CODES_PATH not in sys.path:
    sys.path.insert(0, MM_CODES_PATH)

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=2e7, backupCount=5)
fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
fh.setFormatter(fh_formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(name)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

if not os.path.isdir(LIVE_TMP_PATH):
    os.mkdir(LIVE_TMP_PATH)

app = Flask(__name__)
# app.config.from_object(os.environ['APP_SETTINGS'])
app.config.from_object(APP_SETTINGS)

db = SQLAlchemy(app)



"""
User
"""

def check_token(token):
    key = app.secret_key.encode()
    cipher = AES.new(key, AES.MODE_ECB)
    de_text = base64.decodebytes(token.encode())
    de_text = cipher.decrypt(de_text)
    de_text = Padding.unpad(de_text, 16)
    de_text = de_text.decode()
    username, expire = de_text.split('@')
    expire = datetime.fromtimestamp(float(expire))
    res = db.session.query(User.is_admin).filter(User.user_name==username).first()
    if res is not None and expire > datetime.now():
        is_admin = res.is_admin
        return username, is_admin
    else:
        return None, None


@app.route('/user/login', methods=['POST'])
def login():
    logger.debug("API called: /user/login")
    try:
        username = json.loads(request.get_data())['username']
        password = json.loads(request.get_data())['password']
        res = db.session.query(User.user_id).filter(and_(User.user_name==username, User.password==password))
        if res.first() is not None:
            expire = (datetime.now() + timedelta(hours=24)).timestamp()
            # expire = (datetime.now() + timedelta(seconds=30)).timestamp()
            text = username + '@' + str(expire)
            key = app.secret_key.encode()
            text = Padding.pad(text.encode(), 16)
            cipher = AES.new(key, AES.MODE_ECB)
            en_text = cipher.encrypt(text)
            en_text = base64.encodebytes(en_text)
            en_text = en_text.decode()
            return {'code': SUCCESS_CODE, 'msg': 'success', 'token': en_text}
        else:
            return {'code': SUCCESS_CODE, 'msg': 'Login Failed'}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/user/verifyToken', methods=['POST'])
def verify_token():
    logger.debug("API called: /user/verifyToken")
    try:
        token = json.loads(request.get_data())['token']
        username, is_admin = check_token(token)
        if username is not None:
            return {'code': SUCCESS_CODE, 'msg': 'success', 'is_admin':is_admin, 'user_name': username}
        else:
            return {'code': SUCCESS_CODE, 'msg': 'fail', 'is_admin': False, 'user_name': None}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/user/getUserList', methods=['POST'])
def get_user_list():
    logger.debug("API called: /user/getUserList")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        query = db.session.query(User.user_id, User.user_name, User.is_admin).all()
        res = [u._asdict() for u in query]
        return {'code': SUCCESS_CODE, 'msg': 'success', 'data': res}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/user/addUser', methods=['POST'])
def add_user():
    """
        Add or edit user info.
    """
    logger.debug("API called: /user/addUser")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        username = request_data['username']
        password = request_data['password']
        is_admin = request_data['isAdmin']
        insert_stmt = insert(User).values(
                    user_name = username,
                    password = password,
                    is_admin = is_admin,
                )
        on_duplicate_key_update_stmt = insert_stmt.on_duplicate_key_update(
                    password = password,
                    is_admin = is_admin,
                )
        db.session.execute(on_duplicate_key_update_stmt)
        db.session.commit()
        return {'code': SUCCESS_CODE, 'msg': 'success'}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}



"""
Data-End
"""

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
        logger.error(e)
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
        logger.error(e)
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
        logger.error(e)
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
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "data": res}


@app.route('/dataEnd/getDetails', methods=['POST'])
def get_dataset_details():
    logger.debug("API called: /dataEnd/getDetails")
    try:
        data = json.loads(request.get_data())
        page, page_size = data['pageNo'], data['pageSize']

        samples = db.session.query(Dsample).filter_by(
            dataset_name=data['datasetName'])
        totol_count = samples.count()
        if data['sentiment_filter'] != 'All':
            samples = samples.filter_by(annotation=data['sentiment_filter'])
        if data['data_mode_filter'] != 'All':
            samples = samples.filter_by(data_mode=data['data_mode_filter'])
        if data['id_filter'] != '':
            samples = samples.filter_by(video_id=data['id_filter'])
        samples = samples.limit(page_size).offset((page - 1) * page_size).all()

        res = []
        for sample in samples:
            p = sample.__dict__.copy()
            p.pop('_sa_instance_state', None)
            for k,v in p.items():
                if v == None:
                    p[k] = '-'
            p['video_url'] = os.path.join(DATASET_SERVER_IP, p['video_path'])
            res.append(p)

    except Exception as e:
        logger.error(e)
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
        logger.error(e)
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
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}



"""
DATA-Labeling
"""

@app.route('/dataEnd/getUsersForAssignment', methods=['POST'])
def get_users_for_assignment():
    logger.debug("API called: /dataEnd/getUsersForAssignment")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = request_data['datasetName']
        query_user = db.session.query(User.user_name)
        users = [u.user_name for u in query_user.all()]
        if 'admin' in users:
            users.remove('admin')
        assigned = []
        for user in users:
            query = db.session.query(Annotation).filter(and_(
                Annotation.user_name == user,
                Annotation.dataset_name == dataset_name,
            ))
            assigned.append(query.count())
        res = []
        for i, user in enumerate(users):
            res.append({
                'num': i,
                'username': user,
                'assigned': assigned[i],
            })
        return {"code": SUCCESS_CODE, "msg": 'success', "data": res}
        
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/assignTasks', methods=['POST'])
def assign_tasks():
    logger.debug("API called: /dataEnd/assignTasks")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')
        
        dataset_name = json.loads(request.get_data())['dataset_name']
        users = json.loads(request.get_data())['users']
        res = db.session.query(Dsample.video_id, Dsample.clip_id).filter_by(dataset_name=dataset_name)
        video_ids = [v.video_id for v in res.all()]
        clip_ids = [c.clip_id for c in res.all()]
        for user in users:
            for i, v_id in enumerate(video_ids):
                db.session.execute(f"""
                                    INSERT IGNORE INTO `Annotation` 
                                    (`user_name`, `dataset_name`, `video_id`, `clip_id`)
                                    VALUES ('{user}', '{dataset_name}', '{v_id}', '{clip_ids[i]}');
                                    """)
        db.session.commit()
        return {"code": SUCCESS_CODE, "msg": 'success'}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/calculateLables', methods=['POST'])
def calculate_lables():
    logger.debug("API called: /dataEnd/calculateLables")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = request_data['datasetName']
        threshold = request_data['threshold']
        query_0 = db.session.query (Dsample.video_id, Dsample.clip_id).filter(
            Dsample.dataset_name == dataset_name
        ).all()
        for row in tqdm(query_0):
            v_id, c_id = row
            query_1 = db.session.query(
                Annotation.label_T, Annotation.label_A, Annotation.label_V,Annotation.label_M
            ).filter(and_(
                Annotation.dataset_name == dataset_name,
                Annotation.video_id == v_id,
                Annotation.clip_id == c_id,
            ))
            df = pd.read_sql(query_1.statement, query_1.session.bind)
            avg = df.mean()
            res = []
            for label in df:
                if df[label].count() >= threshold:
                    res.append(avg[label])
                else:
                    res.append(None)
            query_2 = db.session.query(Dsample).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == v_id,
                Dsample.clip_id == c_id,
            )).update({
                'label_T': res[0], 
                'label_A': res[1], 
                'label_V': res[2], 
                'label_value': res[3]
            }, synchronize_session=False)
        db.session.commit()
        return {"code": SUCCESS_CODE, "msg": 'success'}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/exportUserLabels', methods=['POST'])
def export_user_labels():
    logger.debug("API called: /dataEnd/exportUserLabels")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = request_data['datasetName']
        # TODO

    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getLabelingDetails', methods=['POST'])
def get_labeling_details():
    logger.debug("API called: /dataEnd/getLabelingDetails")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        dataset_name = request_data['datasetName']
        id_filter = request_data['idFilter']
        pageNo = request_data['pageNo']
        pageSize = request_data['pageSize']
        samples = db.session.query(
            Annotation.video_id, Annotation.clip_id
        ).filter_by(
            dataset_name=dataset_name
        ).group_by(
            Annotation.video_id, Annotation.clip_id
        )
        if id_filter != '':
            samples = samples.filter(Annotation.video_id == id_filter)
        total = samples.count()
        samples = samples.paginate(pageNo, pageSize, False)
        res = []
        for sample in samples.items:
            result = db.session.query(Annotation).filter(
                and_(
                    Annotation.dataset_name == dataset_name,
                    Annotation.video_id == sample.video_id,
                    Annotation.clip_id == sample.clip_id,
                )
            )
            assigned = result.count()
            label_T = result.filter(Annotation.label_T != None).count()
            label_A = result.filter(Annotation.label_A != None).count()
            label_V = result.filter(Annotation.label_V != None).count()
            label_M = result.filter(Annotation.label_M != None).count()
            res.append({
                'video_id': sample.video_id,
                'clip_id': sample.clip_id,
                'assigned': assigned,
                'label_T': label_T,
                'label_A': label_A,
                'label_V': label_V,
                'label_M': label_M,
                'status': 'Finished' if assigned == label_T == label_A == label_V == label_M else 'Labeling'
            })
        return {"code": SUCCESS_CODE, "msg": 'success', "data": res, "totalCount": total}

    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getMyProgress', methods=['POST'])
def get_my_progress():
    logger.debug("API called: /dataEnd/getMyProgress")
    try:
        token = json.loads(request.get_data())['token']
        username, _ = check_token(token)
        dataset_name = json.loads(request.get_data())['datasetName']
        all = db.session.query(Annotation).filter(and_(
            Annotation.user_name == username, 
            Annotation.dataset_name == dataset_name
        )).count()
        text = db.session.query(Annotation).filter(and_(
            Annotation.user_name == username, 
            Annotation.dataset_name == dataset_name,
            Annotation.label_T != None
        )).count()
        audio = db.session.query(Annotation).filter(and_(
            Annotation.user_name == username, 
            Annotation.dataset_name == dataset_name,
            Annotation.label_A != None
        )).count()
        video = db.session.query(Annotation).filter(and_(
            Annotation.user_name == username, 
            Annotation.dataset_name == dataset_name,
            Annotation.label_V != None
        )).count()
        multi = db.session.query(Annotation).filter(and_(
            Annotation.user_name == username, 
            Annotation.dataset_name == dataset_name,
            Annotation.label_M != None
        )).count()
        res = {
            'all': all,
            'text': text,
            'audio': audio,
            'video': video,
            'multi': multi
        }
        return {"code": SUCCESS_CODE, "msg": 'success', 'data': res}
        
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getAllProgress', methods=['POST'])
def get_all_progress():
    logger.debug("API called: /dataEnd/getAllProgress")
    try:
        token = json.loads(request.get_data())['token']
        username, is_admin = check_token(token)
        dataset_name = json.loads(request.get_data())['datasetName']
        users = []
        label_T = []
        label_A = []
        label_V = []
        label_M = []
        res_users = db.session.query(Annotation.user_name).distinct().all()
        for row in res_users:
            users.append(row.user_name)
            label_T.append(db.session.query(Annotation).filter(and_(
                Annotation.user_name == row.user_name,
                Annotation.dataset_name == dataset_name,
                Annotation.label_T != None,
            )).count())
            label_A.append(db.session.query(Annotation).filter(and_(
                Annotation.user_name == row.user_name,
                Annotation.dataset_name == dataset_name,
                Annotation.label_A != None,
            )).count())
            label_V.append(db.session.query(Annotation).filter(and_(
                Annotation.user_name == row.user_name,
                Annotation.dataset_name == dataset_name,
                Annotation.label_V != None,
            )).count())
            label_M.append(db.session.query(Annotation).filter(and_(
                Annotation.user_name == row.user_name,
                Annotation.dataset_name == dataset_name,
                Annotation.label_M != None,
            )).count())
        
        data = {
            'users': users,
            'label_T': label_T,
            'label_A': label_A,
            'label_V': label_V,
            'label_M': label_M,
        }
        return { "code": SUCCESS_CODE, "msg": 'success', "data": data}

    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getTextSample', methods=['POST'])
def get_text_samples():
    """
        Get next unlabeled text sample.
    """
    logger.debug("API called: /dataEnd/getTextSample")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        # current_video_id = request_data['currentVideoID']
        # current_clip_id = request_data['currentClipID']
        res = db.session.query(Annotation.video_id, Annotation.clip_id).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
                Annotation.label_T == None
            )
        ).limit(1).first()
        if res:
            video_id, clip_id = res
            text = db.session.query(Dsample.text).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first().text
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'transcript': text
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None }
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getTextSampleNext', methods=['POST'])
def get_text_samples_next():
    """
        Get next or previous text sample
    """
    logger.debug("API called: /dataEnd/getTextSamplePrev")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        mode = request_data['mode']
        res = db.session.query(Annotation.id, Annotation.video_id, Annotation.clip_id, Annotation.label_T).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
            )
        )
        res2 = res.filter(and_(
            Annotation.video_id == current_video_id, 
            Annotation.clip_id == current_clip_id
        ))
        current_id = res2.first().id
        if mode == 'prev':
            res3 = res.filter(Annotation.id < current_id).order_by(Annotation.id.desc()).limit(1).first()
        elif mode == 'next':
            res3 = res.filter(Annotation.id > current_id).order_by(Annotation.id.asc()).limit(1).first()
        else:
            raise RuntimeError(f"Invalid argument 'mode': {mode}")
        if res3:
            prev_id = res3.id
            _, video_id, clip_id, label = res.filter(
                Annotation.id == prev_id).first()
            text = db.session.query(Dsample.text).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first().text
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'transcript': text,
                'label_T': label
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/submitTextLabel', methods=["POST"])
def submit_text_label():
    logger.debug("API called: /dataEnd/submitTextLabel")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        label = request_data['label']
        
        db.session.query(Annotation).filter(
            Annotation.user_name == username,
            Annotation.dataset_name == dataset_name,
            Annotation.video_id == current_video_id,
            Annotation.clip_id == current_clip_id
        ).update({'label_T': label}, synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/getAudioSample', methods=['POST'])
def get_audio_samples():
    """
        Get next unlabeled audio sample.
    """
    logger.debug("API called: /dataEnd/getAudioSample")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        res = db.session.query(Annotation.video_id, Annotation.clip_id).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
                Annotation.label_A == None
            )
        ).limit(1).first()
        if res:
            video_id, clip_id = res
            with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
                dataset_config = json.load(fp)
            audio_dir = dataset_config[dataset_name]['audio_for_label_dir']
            audio_format = dataset_config[dataset_name]['audio_format']
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'audio_url': os.path.join(DATASET_SERVER_IP, audio_dir, video_id, clip_id + '.' + audio_format)
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None }
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getAudioSampleNext', methods=['POST'])
def get_audio_samples_next():
    """
        Get next or previous audio sample
    """
    logger.debug("API called: /dataEnd/getAudioSamplePrev")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        mode = request_data['mode']
        res = db.session.query(Annotation.id, Annotation.video_id, Annotation.clip_id, Annotation.label_A).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
            )
        )
        res2 = res.filter(and_(
            Annotation.video_id == current_video_id, 
            Annotation.clip_id == current_clip_id
        ))
        current_id = res2.first().id
        if mode == 'prev':
            res3 = res.filter(Annotation.id < current_id).order_by(Annotation.id.desc()).limit(1).first()
        elif mode == 'next':
            res3 = res.filter(Annotation.id > current_id).order_by(Annotation.id.asc()).limit(1).first()
        else:
            raise RuntimeError(f"Invalid argument 'mode': {mode}")
        if res3:
            prev_id = res3.id
            _, video_id, clip_id, label = res.filter(
                Annotation.id == prev_id).first()
            with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
                dataset_config = json.load(fp)
            audio_dir = dataset_config[dataset_name]['audio_for_label_dir']
            audio_format = dataset_config[dataset_name]['audio_format']
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'audio_url': os.path.join(DATASET_SERVER_IP, audio_dir, video_id, clip_id + '.' + audio_format),
                'label_A': label
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/submitAudioLabel', methods=["POST"])
def submit_audio_label():
    logger.debug("API called: /dataEnd/submitAudioLabel")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        label = request_data['label']
        
        db.session.query(Annotation).filter(
            Annotation.user_name == username,
            Annotation.dataset_name == dataset_name,
            Annotation.video_id == current_video_id,
            Annotation.clip_id == current_clip_id
        ).update({'label_A': label}, synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/getVideoSample', methods=['POST'])
def get_video_samples():
    """
        Get next unlabeled video sample.
    """
    logger.debug("API called: /dataEnd/getVideoSample")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        res = db.session.query(Annotation.video_id, Annotation.clip_id).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
                Annotation.label_V == None
            )
        ).limit(1).first()
        if res:
            video_id, clip_id = res
            video_path = db.session.query(Dsample.video_path).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first().video_path
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'video_url': os.path.join(DATASET_SERVER_IP, video_path)
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None }
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getVideoSampleNext', methods=['POST'])
def get_video_samples_next():
    """
        Get next or previous video sample
    """
    logger.debug("API called: /dataEnd/getVideoSamplePrev")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        mode = request_data['mode']
        res = db.session.query(Annotation.id, Annotation.video_id, Annotation.clip_id, Annotation.label_V).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
            )
        )
        res2 = res.filter(and_(
            Annotation.video_id == current_video_id, 
            Annotation.clip_id == current_clip_id
        ))
        current_id = res2.first().id
        if mode == 'prev':
            res3 = res.filter(Annotation.id < current_id).order_by(Annotation.id.desc()).limit(1).first()
        elif mode == 'next':
            res3 = res.filter(Annotation.id > current_id).order_by(Annotation.id.asc()).limit(1).first()
        else:
            raise RuntimeError(f"Invalid argument 'mode': {mode}")
        if res3:
            prev_id = res3.id
            _, video_id, clip_id, label = res.filter(
                Annotation.id == prev_id).first()
            video_path = db.session.query(Dsample.video_path).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first().video_path
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'video_url': os.path.join(DATASET_SERVER_IP, video_path),
                'label_V': label
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/submitVideoLabel', methods=["POST"])
def submit_video_label():
    logger.debug("API called: /dataEnd/submitVideoLabel")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        label = request_data['label']
        
        db.session.query(Annotation).filter(
            Annotation.user_name == username,
            Annotation.dataset_name == dataset_name,
            Annotation.video_id == current_video_id,
            Annotation.clip_id == current_clip_id
        ).update({'label_V': label}, synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/getMultiSample', methods=['POST'])
def get_multi_samples():
    """
        Get next unlabeled multimodal sample.
    """
    logger.debug("API called: /dataEnd/getMultiSample")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        res = db.session.query(Annotation.video_id, Annotation.clip_id).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
                Annotation.label_M == None
            )
        ).limit(1).first()
        if res:
            video_id, clip_id = res
            video_path, text = db.session.query(Dsample.video_path, Dsample.text).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first()
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'transcript': text,
                'video_url': os.path.join(DATASET_SERVER_IP, video_path)
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None }
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/getMultiSampleNext', methods=['POST'])
def get_multi_samples_next():
    """
        Get next or previous multimodal sample
    """
    logger.debug("API called: /dataEnd/getMultiSamplePrev")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        mode = request_data['mode']
        res = db.session.query(Annotation.id, Annotation.video_id, Annotation.clip_id, Annotation.label_M).filter(
            and_(
                Annotation.user_name == username, 
                Annotation.dataset_name == dataset_name, 
            )
        )
        res2 = res.filter(and_(
            Annotation.video_id == current_video_id, 
            Annotation.clip_id == current_clip_id
        ))
        current_id = res2.first().id
        if mode == 'prev':
            res3 = res.filter(Annotation.id < current_id).order_by(Annotation.id.desc()).limit(1).first()
        elif mode == 'next':
            res3 = res.filter(Annotation.id > current_id).order_by(Annotation.id.asc()).limit(1).first()
        else:
            raise RuntimeError(f"Invalid argument 'mode': {mode}")
        if res3:
            prev_id = res3.id
            _, video_id, clip_id, label = res.filter(
                Annotation.id == prev_id).first()
            video_path, text = db.session.query(Dsample.video_path, Dsample.text).filter(and_(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == video_id,
                Dsample.clip_id == clip_id
            )).first()
            data = {
                'dataset_name':dataset_name,
                'video_id': video_id,
                'clip_id': clip_id,
                'transcript': text,
                'video_url': os.path.join(DATASET_SERVER_IP, video_path),
                'label_M': label
            }
            return { "code": SUCCESS_CODE, "msg": 'success', "data": data }
        else:
            return { "code": SUCCESS_CODE, "msg": 'no more', "data": None}
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}


@app.route('/dataEnd/submitMultiLabel', methods=["POST"])
def submit_multi_label():
    logger.debug("API called: /dataEnd/submitMultiLabel")
    try:
        request_data = json.loads(request.get_data())
        token = request_data['token']
        username, _ = check_token(token)
        dataset_name = request_data['datasetName']
        current_video_id = request_data['currentVideoID']
        current_clip_id = request_data['currentClipID']
        label = request_data['label']
        
        db.session.query(Annotation).filter(
            Annotation.user_name == username,
            Annotation.dataset_name == dataset_name,
            Annotation.video_id == current_video_id,
            Annotation.clip_id == current_clip_id
        ).update({'label_M': label}, synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/exportLabels', methods=['POST'])
def export_labels():
    logger.debug("API called: /dataEnd/exportLabels")
    # TODO: rewrite this function
    try:
        dataset_name = json.loads(request.get_data())['datasetName']
        samples = db.Query(Dsample).filter_by(dataset_name=dataset_name).all()
        # {"video_id$_$clip_id": [label, label_by, annotation]}
        name_label_dict = {}
        for sample in samples:
            key = f'{sample.video_id}$_${sample.clip_id}'
            name_label_dict[key] = [sample.label_value,
                                    sample.label_by, sample.annotation]
        # load label file
        with open(os.path.join(AL_CODES_PATH, 'config.json'), 'r') as fp:
            config = json.load(fp)
        label_path = config['data'][dataset_name]['label_path']
        df = pd.read_csv(label_path, encoding='utf-8',
                         dtype={'video_id': str, 'clip_id': str})
        new_labels, new_label_bys, new_annotations = [], [], []
        for i in range(len(df)):
            video_id, vlip_id = df.loc[i, ['video_id', 'clip_id']]
            key = f'{video_id}$_${clip_id}'
            new_labels.append(name_label_dict[key][0])
            new_label_bys.append(name_label_dict[key][1])
            new_annotations.append(name_label_dict[key][2])
        # update label file
        df['label'] = new_labels
        df['label_by'] = new_label_bys
        df['annotation'] = new_annotations
        df.to_csv(label_path, index=None, encoding='utf-8')
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


"""
Feature-End
"""


@app.route('/featureEnd/getFeatureList', methods=['POST'])
def get_feature_list():
    logger.debug("API called: /featureEnd/getFeatureList")
    try:
        request_data = json.loads(request.get_data())
        dataset_name = request_data['datasetName']
        feature_name = request_data['featureName']
        feature_T = request_data['featureT']
        feature_A = request_data['featureA']
        feature_V = request_data['featureV']
        query_result = db.session.query(
            Feature.id, Feature.feature_name, Feature.dataset_name, Feature.feature_T,
            Feature.feature_A, Feature.feature_V, Feature.description
        )
        for filter_item in ['dataset_name', 'feature_T', 'feature_A', 'feature_V']:
            if eval(filter_item) != '':
                query_result = query_result.filter(getattr(Feature, filter_item) == eval(filter_item))
        if feature_name != '':
            query_result = query_result.filter(Feature.feature_name.like(f'%{feature_name}%'))
        res = []
        for result in query_result:
            res.append(result._asdict())
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": res, "total": len(res)}


@app.route('/featureEnd/scanDefaultFeatures', methods=['POST'])
def scan_default_features():
    logger.debug("API called: /featureEnd/scanDefaultFeatures")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')

        # load config file
        with open(os.path.join(DATASET_ROOT_DIR, 'config.json'), 'r') as fp:
            config = json.load(fp)
        for dataset_name in config:
            try:
                for feature_type in config[dataset_name]['features']:
                    feature_name = 'Default_' + feature_type
                    try:
                        feature_config = config[dataset_name]['features'][feature_type]
                        if feature_config['feature_path'] == '':
                            continue
                        feature_path = os.path.join(DATASET_ROOT_DIR, feature_config['feature_path'])
                        feature_description = feature_config['description'] if 'description' in feature_config else None
                        payload = Feature(
                            dataset_name=dataset_name,
                            feature_name=feature_name,
                            feature_path=feature_path,
                            description=feature_description
                        )
                        db.session.add(payload)
                    except Exception:
                        continue
            except Exception:
                continue
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/removeInvalidFeatures', methods=['GET'])
def remove_invalid_features():
    logger.debug("API called: /featureEnd/removeInvalidFeatures")
    try:
        token = json.loads(request.get_data())['token']
        _, is_admin = check_token(token)
        if is_admin != True:
            raise RuntimeError('Authentication Error')
        
        query_result = db.session.query(Feature.id, Feature.feature_path)
        for row in query_result:
            if not os.path.isfile(row.feature_path):
                db.session.query(Feature).filter_by(id=row.id).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/removeFeatures', methods=['POST'])
def remove_features():
    logger.debug("API called: /featureEnd/removeFeatures")
    try:
        request_data = json.loads(request.get_data())
        ids = request_data['id']
        db.session.query(Feature).filter(Feature.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/startExtracting', methods=['POST'])
def start_extracting():
    logger.debug("API called: /featureEnd/startExtracting")
    try:
        request_data = json.loads(request.get_data())
        # TODO
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


"""
Model-End
"""


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
        logger.error(e)
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
            '--description', data['description']
        ]
        p = subprocess.Popen(cmd_page, close_fds=True)

        payload.task_pid = p.pid
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/settings/getAllSettings', methods=['GET'])
def get_settings():
    logger.debug("API called: /modelEnd/getAllSettings")
    try:
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
            cur_data['sentiments'] = [
                v for k, v in dataset['annotations'].items()]
            ret_datas.append(cur_data)

        pre_trained_models = []
        defaults = db.session.query(Result).filter_by(is_tuning='Train').all()
        for default in defaults:
            cur_name = default.model_name + '-' + \
                default.dataset_name + '-' + str(default.result_id)
            pre_trained_models.append(cur_name)
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "models": model_names,
            "datasets": ret_datas, "pretrained": pre_trained_models}


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
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "args": json.dumps(args)}


"""
Analysis-End
"""


@app.route('/analysisEnd/getResults', methods=["POST"])
def get_results():
    logger.debug("API called: /analysisEnd/getResults")
    try:
        data = json.loads(request.get_data())

        results = db.session.query(Result)
        if data['model_name'] != 'All':
            results = results.filter_by(model_name=data['model_name'])
        if data['dataset_name'] != 'All':
            results = results.filter_by(dataset_name=data['dataset_name'])
        if data['is_tuning'] != 'Both':
            results = results.filter_by(is_tuning=data['is_tuning'])

        # sorted results
        if data['order'] == 'descending':
            results = results.order_by(
                eval('Result.'+data['sortBy']).desc()).all()
        elif data['order'] == 'ascending':
            results = results.order_by(
                eval('Result.'+data['sortBy']).asc()).all()
        else:
            results = results.all()

        ret = []

        for result in results:
            p = result.__dict__.copy()
            p.pop('_sa_instance_state', None)
            cur_id = p['result_id']
            p['created_at'] = p['created_at'].astimezone(
                timezone('Asia/Shanghai'))
            p['test-acc'] = result.accuracy
            p['test-f1'] = result.f1
            p['train'] = {k: [] for k in ['loss_value', 'accuracy', 'f1']}
            p['valid'] = {k: [] for k in ['loss_value', 'accuracy', 'f1']}
            p['test'] = {k: [] for k in ['loss_value', 'accuracy', 'f1']}
            e_result = db.session.query(EResult).filter_by(
                result_id=result.result_id).order_by(asc(EResult.epoch_num)).all()
            e_result = e_result[1:]  # remove final results
            for cur_r in e_result:
                e_res = json.loads(cur_r.results)
                for mode in ['train', 'valid', 'test']:
                    for item in ['loss_value', 'accuracy', 'f1']:
                        p[mode][item].append(e_res[mode][item])
            ret.append(p)

        page, pageSize = data['pageNo'], data['pageSize']

        totolCount = len(ret)
        start_i = (page - 1) * pageSize
        if start_i > totolCount:
            return {"code": ERROR_CODE, "msg": 'page error!'}
        end_i = (start_i + pageSize) if (start_i +
                                         pageSize) <= totolCount else totolCount
        ret = ret[start_i:end_i]
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "results": ret}


@app.route('/analysisEnd/getResultDetails', methods=["POST"])
def get_results_details():
    logger.debug("API called: /analysisEnd/getResultDetails")
    try:
        result_id = json.loads(request.get_data())['id']
        # print(result_id)
        cur_result = db.session.query(Result).get(result_id)
        ret = {
            "id": result_id,
            "model": cur_result.model_name,
            "dataset": cur_result.dataset_name,
            "args": cur_result.args,
            "description": cur_result.description,
            "train": {k: [] for k in ['loss_value', 'accuracy', 'f1']},
            "valid": {k: [] for k in ['loss_value', 'accuracy', 'f1']},
            "test": {k: [] for k in ['loss_value', 'accuracy', 'f1']},
            "features": {}
        }
        e_result = db.session.query(EResult).filter_by(
            result_id=cur_result.result_id).order_by(asc(EResult.epoch_num)).all()
        for cur_e in e_result:
            e_res = json.loads(cur_e.results)
            for mode in ['train', 'valid', 'test']:
                for item in ['loss_value', 'accuracy', 'f1']:
                    ret[mode][item].append(round(e_res[mode][item], 4))
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "results": ret}


@app.route('/analysisEnd/getFeatureDetails', methods=["POST"])
def get_feature_details():
    logger.debug("API called: /analysisEnd/getFeatureDetails")
    try:
        data = json.loads(request.get_data())
        result_id, select_modes, feature_mode = data['id'], data['select_modes'], data['feature_mode']
        cur_result = db.session.query(Result).get(result_id)
        ret = {}
        if cur_result.is_tuning == 'Train':
            # load features
            with open(os.path.join(MODEL_TMP_SAVE, f'{cur_result.model_name}-{cur_result.dataset_name}-{result_id}.pkl'), 'rb') as fp:
                features = pickle.load(fp)
            select_modes = [s.lower() for s in select_modes]
            final_select_modes = []
            if 'train' in select_modes:
                final_select_modes.append('train')
            if 'valid' in select_modes:
                final_select_modes.append('valid')
            if 'test' in select_modes:
                final_select_modes.append('test')
            key = '-'.join(final_select_modes)
            for name in ['Feature_T', 'Feature_A', 'Feature_V', 'Feature_M']:
                ret[name] = features[key][feature_mode][name]
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "features": ret}


@app.route('/analysisEnd/getSampleDetails', methods=["POST"])
def get_sample_details():
    logger.debug("API called: /analysisEnd/getSampleDetails")
    try:
        data = json.loads(request.get_data())
        result_mode, data_mode = data['result_mode'], data['data_mode']

        sRs = db.session.query(SResults).filter_by(result_id=data['id']).all()
        ret = []
        for s_res in sRs:
            if result_mode == "All" or \
                (result_mode == "Right" and s_res.label_value == s_res.predict_value) or \
                    (result_mode == "Wrong" and s_res.label_value != s_res.predict_value):
                cur_sample = db.session.query(Dsample).get(s_res.sample_id)
                if cur_sample and (data['video_id'] == '' or cur_sample.video_id == data['video_id']):
                    if data_mode == "All" or (cur_sample.data_mode == data_mode.lower()):
                        cur_res = {
                            "sample_id": s_res.sample_id,
                            "video_id": cur_sample.video_id,
                            "clip_id": cur_sample.clip_id,
                            "data_mode": cur_sample.data_mode,
                            "predict_value": s_res.predict_value,
                            "label_value": s_res.label_value,
                            "text": cur_sample.text,
                            "video_url": os.path.join(DATASET_SERVER_IP, cur_sample.video_path)
                        }
                        ret.append(cur_res)

        page, pageSize = data['pageNo'], data['pageSize']

        totolCount = len(ret)
        start_i = (page - 1) * pageSize
        if start_i > totolCount:
            return {"code": ERROR_CODE, "msg": 'page error!'}
        end_i = (start_i + pageSize) if (start_i +
                                         pageSize) <= totolCount else totolCount
        ret = ret[start_i:end_i]
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "results": ret}


@app.route('/analysisEnd/setDefaultModel', methods=['POST'])
def set_default_args():
    logger.debug("API called: /analysisEnd/setDefaultModel")
    try:
        result_id = json.loads(request.get_data())['id']
        cur_result = db.session.query(Result).get(result_id)
        # revise config.json in model codes
        with open(os.path.join(MM_CODES_PATH, 'config.json'), 'r') as f:
            model_config = json.load(f)
        model_config['MODELS'][cur_result.model_name]['args'][cur_result.dataset_name] = json.loads(
            cur_result.args)
        with open(os.path.join(MM_CODES_PATH, 'config.json'), 'w') as f:
            json.dump(model_config, f, indent=4)
        # cur_result.is_default = True
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/analysisEnd/delResult', methods=['POST'])
def del_results():
    logger.debug("API called: /analysisEnd/delResult")
    try:
        result_ids = json.loads(request.get_data())['id']
        # print(result_ids)
        for result_id in result_ids:
            cur_result = db.session.query(Result).get(result_id)
            cur_name = cur_result.model_name + '-' + \
                cur_result.dataset_name + '-' + str(cur_result.result_id)
            file_paths = glob(os.path.join(MODEL_TMP_SAVE, cur_name + '.*'))
            for file in file_paths:
                os.remove(file)

            db.session.delete(cur_result)
            e_results = db.session.query(EResult).filter_by(
                result_id=result_id).all()
            for e_result in e_results:
                db.session.delete(e_result)
            s_results = db.session.query(SResults).filter_by(
                result_id=result_id).all()
            for s_result in s_results:
                db.session.delete(s_result)
            db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/analysisEnd/batchResults', methods=['POST'])
def batch_test():
    logger.debug("API called: /analysisEnd/batchResults")
    try:
        data = json.loads(request.get_data())
        models = data['model']
        mode = data['mode'].lower()
        results = []
        for model in models:
            result_id = int(model.split('-')[-1])
            cur_result = {k: [] for k in ['loss_value', 'accuracy', 'f1']}
            cur_result['model'] = model
            e_result = db.session.query(EResult).filter_by(
                result_id=result_id).order_by(asc(EResult.epoch_num)).all()
            for cur_e in e_result:
                e_res = json.loads(cur_e.results)
                for item in ['loss_value', 'accuracy', 'f1']:
                    cur_result[item].append(round(e_res[mode][item], 4))
            results.append(cur_result)
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {'code': SUCCESS_CODE, 'msg': 'success', 'result': results}


@app.route('/analysisEnd/runLive', methods=['POST'])
def get_live_results():
    logger.debug("API called: /analysisEnd/runLive")
    try:
        # print(request.form)
        msc = str(round(time.time() * 1000))
        working_dir = os.path.join(LIVE_TMP_PATH, msc)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        # save video
        with open(os.path.join(working_dir, "live.mp4"), "wb") as vid:
            video_stream = request.files['recorded'].stream.read()
            # video_stream = request.form['video'].stream.read()
            vid.write(video_stream)
        # load other params
        pre_trained_models = request.form['model'].split(',')
        results = {k: [] for k in "MTAV"}
        for pre_trained_model in pre_trained_models:
            model_name, dataset_name = pre_trained_model.split('-')[0:2]
            other_args = {
                'pre_trained_model': pre_trained_model + '.pth',
                'modelName': model_name,
                'datasetName': dataset_name,
                'live_working_dir': working_dir,
                'transcript': request.form['transcript'],
                'language': request.form['language']
            }
            cur_results = run_live(other_args)
            for k, v in cur_results.items():
                v['model'] = pre_trained_model
                results[k].append(v)
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    finally:
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
    return {"code": SUCCESS_CODE, "msg": "success", "result": results}


"""
Tasks
"""


@app.route('/task/getTaskList', methods=["GET"])
def get_task_list():
    logger.debug("API called: /task/getTaskList")
    try:
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
            p['start_time'] = p['start_time'].astimezone(
                timezone('Asia/Shanghai'))
            p['end_time'] = p['end_time'].astimezone(timezone('Asia/Shanghai'))
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
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": ret}


@app.route('/task/stopTask', methods=['POST'])
def stop_task():
    logger.debug("API called: /task/stopTask")
    try:
        data = json.loads(request.get_data())
        task_id = data['task_id']
        cur_task = db.session.query(Task).get(task_id)

        cmd = 'kill -9 ' + str(cur_task.task_pid)
        os.system(cmd)

        cur_task.state = 3
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/task/delTask', methods=['POST'])
def delete_task():
    logger.debug("API called: /task/delTask")
    try:
        data = json.loads(request.get_data())
        # print(data)
        task_id = data['task_id']
        cur_task = db.session.query(Task).get(task_id)
        db.session.delete(cur_task)
        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/task/delAllTask', methods=["GET"])
def del_unrun_tasks():
    logger.debug("API called: /task/delAllTask")
    try:
        tasks = db.session.query(Task).all()
        for task in tasks:
            if task.state != 0:
                db.session.delete(task)

        db.session.commit()
    except Exception as e:
        logger.error(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}



@app.after_request
def after(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = request.headers.get(
        'Origin') or 'http://127.0.0.1:1024'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    # resp.headers['Access-Control-Allow-Credentials'] = 'true'
    return resp


if __name__ == "app":
    logger.info("========================= Program Started =========================")
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)

if __name__ == "__main__":
    logger.info("========================= Program Started =========================")
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)
    app.run(host='0.0.0.0', port=8000)
