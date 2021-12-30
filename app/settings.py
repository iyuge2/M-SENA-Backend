import json
import logging
import os

from constants import *
from database import Result
from MSA_FET import *

from app import app, db


logger = logging.getLogger('app')


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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "models": model_names,
            "datasets": ret_datas, "pretrained": pre_trained_models}


@app.route('/settings/getFeatureExtractionTools', methods=['GET'])
def get_feature_extraction_tools():
    logger.debug("API called: /modelEnd/getFeatureExtractionTools")
    try:
        res = {}
        res['audio'] = list(AUDIO_EXTRACTOR_MAP.keys())
        res['visual'] = list(VIDEO_EXTRACTOR_MAP.keys())
        res['text'] = list(TEXT_EXTRACTOR_MAP.keys())
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": res}