import json
import logging
import os
import pickle
import shutil
import time
from glob import glob

from config.constants import *
from database import Dsample, EResult, Result, SResults
from flask import request
from MMSA.run_live import run_live
from pytz import timezone
from sqlalchemy import and_, asc

from app import app, db
from app.user import check_token


logger = logging.getLogger('app')


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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "features": ret}


@app.route('/analysisEnd/getSampleDetails', methods=["POST"])
def get_sample_details():
    logger.debug("API called: /analysisEnd/getSampleDetails")
    try:
        data = json.loads(request.get_data())
        result_mode, data_mode = data['result_mode'], data['data_mode']
        page, pageSize = data['pageNo'], data['pageSize']
        video_id = data['video_id']

        query_results = db.session.query(SResults, Dsample).filter(
            and_(
                SResults.result_id == data['id'],
                SResults.sample_id == Dsample.sample_id,
            )
        )
        if video_id != '':
            query_results = query_results.filter(Dsample.video_id == video_id)
        if result_mode == 'Right':
            query_results = query_results.filter(SResults.label_value == SResults.predict_value)
        elif result_mode == 'Wrong':
            query_results = query_results.filter(SResults.label_value != SResults.predict_value)
        if data_mode != 'All':
            query_results = query_results.filter(Dsample.data_mode == data_mode.lower())
        totolCount = query_results.count()
        query_results = query_results.paginate(page, pageSize, False)
        res = []
        for sample in query_results.items:
            res.append({
                "sample_id": sample.SResults.sample_id,
                "video_id": sample.Dsample.video_id,
                "clip_id": sample.Dsample.clip_id,
                "data_mode": sample.Dsample.data_mode,
                "predict_value": sample.SResults.predict_value,
                "label_value": sample.SResults.label_value,
                "text": sample.Dsample.text,
                "video_url": os.path.join(DATASET_SERVER_IP, sample.Dsample.video_path),
            })
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "totalCount": totolCount, "results": res}


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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    finally:
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
    return {"code": SUCCESS_CODE, "msg": "success", "result": results}
