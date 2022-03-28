import json
import logging
import os
from flask.helpers import make_response

import pandas as pd
from config.constants import *
from database import Annotation, Dsample, User
from flask import request, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from tqdm import tqdm

from app import app, db
from app.user import check_token


logger = logging.getLogger('app')


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
        logger.exception(e)
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
        logger.exception(e)
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
        # Query for list of (video_id, clip_id) tuples
        ids = db.session.query (Dsample.video_id, Dsample.clip_id).filter(
            Dsample.dataset_name == dataset_name
        ).all()
        # Query for all labels
        labels = db.session.query(
            Annotation.label_T, Annotation.label_A,
            Annotation.label_V, Annotation.label_M
        )
        # Iterate over samples
        for row in tqdm(ids):
            v_id, c_id = row
            # Get labels for this sample
            label_query = labels.filter(
                Annotation.dataset_name == dataset_name,
                Annotation.video_id == v_id,
                Annotation.clip_id == c_id,
            )
            # Convert Query to DataFrame
            df = pd.read_sql(label_query.statement, label_query.session.bind)
            avg = df.mean(axis=0) # Average over rows
            res = []
            # Iterate over 4 labels
            for label in df:
                if df[label].count() >= threshold:
                    res.append(avg[label])
                else:
                    res.append(None)
            # Append "annotation"
            if res[3]:
                res.append('Negative' if res[3] < 0 else 'Positive' if res[3] > 0 else 'Neutral')
            else:
                res.append(None)
            # Update database
            db.session.query(Dsample).filter(
                Dsample.dataset_name == dataset_name,
                Dsample.video_id == v_id,
                Dsample.clip_id == c_id,
            ).update({
                'label_T': res[0], 
                'label_A': res[1], 
                'label_V': res[2], 
                'label_value': res[3],
                'annotation': res[4]
            }, synchronize_session=False)
        db.session.commit()
        return {"code": SUCCESS_CODE, "msg": 'success'}
    except Exception as e:
        db.session.rollback()
        logger.exception(e)
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
        ids = db.session.query(Annotation.video_id, Annotation.clip_id).filter(
            Annotation.dataset_name == dataset_name
        ).distinct()
        users = db.session.query(Annotation.user_name).filter(
            Annotation.dataset_name == dataset_name
        ).distinct()
        users = [u.user_name for u in users]
        v_ids = [v.video_id for v in ids]
        c_ids = [c.clip_id for c in ids]
        df = {}
        for m in ['M', 'T', 'A', 'V']:
            df[m] = pd.DataFrame({
                'video_id': v_ids,
                'clip_id': c_ids,
            })
        for m in ['M', 'T', 'A', 'V']:
            for user in users:
                query = db.session.query(
                    Annotation.video_id,
                    Annotation.clip_id,
                    getattr(Annotation, 'label_' + m)
                ).filter(
                    Annotation.dataset_name == dataset_name,
                    Annotation.user_name == user,
                )
                user_df = pd.read_sql(query.statement, query.session.bind).rename(
                    columns={'label_' + m: user}
                )
                df[m] = pd.merge(df[m], user_df, on=['video_id', 'clip_id'])
        file = os.path.join(DATASET_ROOT_DIR, dataset_name, 'user_labels.xlsx')
        writer = pd.ExcelWriter(file)
        for m in ['M', 'T', 'A', 'V']:
            df[m].to_excel(writer, sheet_name=f"label_{m}", index=False)
        writer.save()
        # return send_file(file, as_attachment=True)
        url = DATASET_SERVER_IP + '/' + os.path.join(dataset_name, 'user_labels.xlsx')
        return {"code": SUCCESS_CODE, "msg": 'success', "url": url}
    except Exception as e:
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
            # check if assigned
            assigned = db.session.query(Annotation).filter(
                and_(
                    Annotation.user_name == row.user_name,
                    Annotation.dataset_name == dataset_name,
                    )
                ).count()
            if assigned > 0:
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/exportLabels', methods=['POST'])
def export_labels():
    logger.debug("API called: /dataEnd/exportLabels")
    try:
        dataset_name = json.loads(request.get_data())['datasetName']
        samples = db.session.query(Dsample).filter(
            Dsample.dataset_name == dataset_name
        )
        df = pd.read_sql(samples.statement, samples.session.bind)
        df = df.drop(columns=['sample_id', 'dataset_name', 'video_path'])
        df = df.rename(columns={'data_mode': 'mode', 'label_value': 'label'})
        df = df[['video_id', 'clip_id', 'text', 'label', 'label_T', 'label_A', 'label_V', 'annotation', 'mode']]
        csv_file = os.path.join(DATASET_ROOT_DIR, dataset_name, 'label.csv')
        df.to_csv(csv_file, index=None, encoding='utf-8')
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@app.route('/dataEnd/submitQuestionableSample', methods=['POST'])
def submit_questionable_sample():
    logger.debug("API called: /dataEnd/submitQuestionableSample")
    try:
        request_data = json.loads(request.get_data())
        dataset_name = request_data['datasetName']
        with open(os.path.join(DATASET_ROOT_DIR, dataset_name, 'questionable_samples.txt'), 'a') as f:
            f.write(request_data['sample'] + '\n')
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}