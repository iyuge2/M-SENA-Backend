import json
import logging
import os
from datetime import datetime
from multiprocessing import Process, Queue

from config.constants import DATASET_ROOT_DIR, ERROR_CODE, SUCCESS_CODE
from database import Feature, Task
from flask import request
from MSA_FET import FeatureExtractionTool, get_default_config
from sqlalchemy.exc import IntegrityError

from app import app, db, progress_queue, sockets
from app.user import check_token

logger = logging.getLogger('app')


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
        page_no = request_data['pageNo']
        page_size = request_data['pageSize']
        query_result = db.session.query(Feature)
        for filter_item in ['dataset_name', 'feature_T', 'feature_A', 'feature_V']:
            if eval(filter_item) != '':
                query_result = query_result.filter(getattr(Feature, filter_item) == eval(filter_item))
        if feature_name != '':
            query_result = query_result.filter(Feature.feature_name.like(f'%{feature_name}%'))
        total_count = query_result.count()
        query_result = query_result.paginate(page_no, page_size, False)
        res = []
        for result in query_result.items:
            p = result.__dict__.copy()
            p.pop('_sa_instance_state', None)
            for k,v in p.items():
                if v == None:
                    p[k] = '-'
            res.append(p)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": res, "total": total_count}


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
                            feature_T='Unknown',
                            feature_A='Unknown',
                            feature_V='Unknown',
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/removeInvalidFeatures', methods=['GET'])
def remove_invalid_features():
    logger.debug("API called: /featureEnd/removeInvalidFeatures")
    try:
        query_result = db.session.query(Feature.id, Feature.feature_path)
        for row in query_result:
            if not os.path.isfile(row.feature_path):
                db.session.query(Feature).filter_by(id=row.id).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/removeFeatures', methods=['POST'])
def remove_features():
    logger.debug("API called: /featureEnd/removeFeatures")
    try:
        request_data = json.loads(request.get_data())
        ids = request_data['id']
        rows = db.session.query(Feature).filter(Feature.id.in_(ids))
        for row in rows:
            os.remove(row.feature_path)
            db.session.delete(row)
        # db.session.query(Feature).filter(Feature.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/featureEnd/getFeatureArgs', methods=['POST'])
def get_feature_args():
    logger.debug("API called: /featureEnd/getFeatureArgs")
    try:
        request_data = json.loads(request.get_data())
        modality = request_data['modality']
        tool = request_data['tool']
        config = get_default_config(tool)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": config}


@app.route('/featureEnd/startExtracting', methods=['POST'])
def start_extracting():
    # TODO: 数据库修改为存放特征参数json字符串，（增加长度，取消索引）
    # TODO: 前端修改为折叠显示三模态特征参数
    logger.debug("API called: /featureEnd/startExtracting")
    try:
        request_data = json.loads(request.get_data())
        advanced = request_data['advanced']
        enable = {
            'text': request_data['text'],
            'audio': request_data['audio'],
            'video': request_data['video']
        }
        tool = {
            'text': request_data['textTool'],
            'audio': request_data['audioTool'],
            'video': request_data['videoTool']
        }
        args = {
            'text': request_data['textArgs'],
            'audio': request_data['audioArgs'],
            'video': request_data['videoArgs']
        }
        if advanced:
            feature_name = request_data['featureName']
        else:
            feature_name = request_data['dataset'] + '_'
            if enable['text']:
                feature_name += 'T_'
            if enable['audio']:
                feature_name += 'A_'
            if enable['video']:
                feature_name += 'V_'
            feature_name = feature_name[:-1]
        dataset = request_data['dataset']
        description = request_data['description']
        config = {}
        for k, v in args.items():
            if enable[k]:
                config = config | v

        database = {}
        # TODO: check user input to avoid vulnerability
        if enable['audio']:
            if tool['audio'] == 'librosa':
                database['audio'] = 'librosa&n_mfcc=' + str(config['audio']['args']['mfcc']['n_mfcc'])
                for item in ['rms', 'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']:
                    if item in config['audio']['args']:
                        database['audio'] += f'&{item}'
            elif tool['audio'] == 'opensmile':
                database['audio'] = f'opensmile&feature_set={config["audio"]["args"]["feature_set"]}&feature_level={config["audio"]["args"]["feature_level"]}'
            elif tool['audio'] == 'wav2vec':
                database['audio'] = f'wav2vec&pretrained={config["audio"]["pretrained"]}'
            else:
                database['audio'] = tool['audio']
        if enable['video']:
            if tool['video'] == 'openface':
                database['video'] = 'openface'
                for item in ['action_units', 'gaze', 'head_pose', 'landmark_2D', 'landmark_3D', 'pdmparams']:
                    if config['video']['args'][item]:
                        database['video'] += f'&{item}'
            else:
                database['video'] = tool['video']
        if enable['text']:
            database['text'] = tool['text']
        
        # insert into Feature table
        try:
            payload_feature = Feature(
                dataset_name=dataset,
                feature_name=feature_name,
                feature_T=database['text'] if enable['text'] else 'None',
                feature_A=database['audio'] if enable['audio'] else 'None',
                feature_V=database['video'] if enable['video'] else 'None',
                feature_path='pending',
                description=description
            )
            db.session.add(payload_feature)
            db.session.flush()
        except IntegrityError:
            db.session.rollback()
            res = db.session.query(Feature.id).filter(
                Feature.dataset_name == dataset,
                Feature.feature_name == feature_name,
                Feature.feature_T == (database['text'] if enable['text'] else 'None'),
                Feature.feature_A == (database['audio'] if enable['audio'] else 'None'),
                Feature.feature_V == (database['video'] if enable['video'] else 'None'),
            ).first()
            return {"code": SUCCESS_CODE, "msg": 'success', "exist": True, "id": res.id}
        
        save_file = os.path.join(DATASET_ROOT_DIR, dataset, 'Processed', 'feature_' + str(payload_feature.id) + '.pkl')
        payload_feature.feature_path = save_file

        # insert into Task table
        payload_task = Task(
            dataset_name = dataset,
            model_name = 'MMSA-FET',
            task_type = 4,
            task_pid = 0,
            state = 0,
            start_time = datetime.now()
        )
        db.session.add(payload_task)
        db.session.flush()
        task_id = payload_task.task_id

        # TODO: use Celery to run the task
        p = Process(target=FeatureExtractionTool(config=config, dataset_root_dir=DATASET_ROOT_DIR
        ).run_dataset, kwargs={
            'dataset_name': dataset,
            'out_file': save_file,
            # 'num_workers': 0,
            'progress_q': progress_queue,
            'task_id': task_id,
        })
        p.start()
        payload_task.task_pid = p.pid

        db.session.commit()

    except Exception as e:
        logger.exception(e)
        db.session.rollback()
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "exist": False, "id": task_id}


@app.route('/featureEnd/getFeatureListforTraining', methods=['POST'])
def get_feature_list_for_training():
    logger.debug("API called: /featureEnd/getFeatureListforTraining")
    try:
        request_data = json.loads(request.get_data())
        modality = request_data['modality']
        dataset_name = request_data['dataset']
        result = []
        query_result = db.session.query(
            Feature.id, Feature.feature_name, Feature.feature_path
        ).filter(
            Feature.dataset_name == dataset_name,
            getattr(Feature, 'feature_' + modality) != 'None',
        )
        for item in query_result:
            result.append({
                'name': f"{item.feature_name}-{str(item.id)}",
                'value': item.feature_path
            })

    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": result}
