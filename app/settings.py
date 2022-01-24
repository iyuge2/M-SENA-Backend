import json
import logging
import os

from config.constants import *
from database import Result
from MSA_FET import *
from MMSA import SUPPORTED_MODELS, SUPPORTED_DATASETS

from app import app, db


logger = logging.getLogger('app')


@app.route('/settings/getAllSettings', methods=['GET'])
def get_settings():
    logger.debug("API called: /modelEnd/getAllSettings")
    try:
        trained_models = []
        defaults = db.session.query(Result).filter_by(is_tune=0).all()
        for default in defaults:
            cur_name = default.model_name + '-' + \
                default.dataset_name + '-' + str(default.result_id)
            trained_models.append(cur_name)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "models": SUPPORTED_MODELS,
            "datasets": SUPPORTED_DATASETS, "trained": trained_models}


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