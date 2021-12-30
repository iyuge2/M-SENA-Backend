import json
import logging
import os
import time
from multiprocessing import Process, Queue
from queue import Empty
from flask import request
from pytz import timezone
from datetime import datetime


from constants import *
from database import Task, User

from app import app, db, sockets, progress_queue
from app.user import check_token


logger = logging.getLogger('app')


@app.route('/task/getTaskList', methods=["GET"])
def get_task_list():
    logger.debug("API called: /task/getTaskList")
    try:
        tasks = db.session.query(Task).all()

        task_type_dict = {
            0: 'Machine Labeling',
            1: 'Model Training',
            2: 'Model Tuning',
            3: 'Model Test',
            4: 'Feature Extraction'
        }
        run_tasks, error_tasks, terminate_tasks, finished_tasks = [], [], [], []
        for task in tasks:
            p = task.__dict__.copy()
            p.pop('_sa_instance_state', None)
            p['task_type'] = task_type_dict[p['task_type']]
            p['start_time'] = p['start_time'].astimezone(
                timezone('Asia/Shanghai'))
            # time is naive, astimezone() will make it aware using local time
            # doesn't matter what timezone is used.
            p['end_time'] = p['end_time'].astimezone(timezone('UTC'))
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success', "data": ret}


@app.route('/task/updateTask', methods=['POST'])
def update_task():
    logger.debug("API called: /task/updateTask")
    try:
        request_data = json.loads(request.get_data())
        task_id = request_data['id']
        state = request_data['state']
        end_time = datetime.fromisoformat(request_data['endTime'])
        row = db.session.query(Task).filter(Task.task_id == task_id).update({
            'state': state,
            'end_time': end_time
        })
        db.session.commit()
        return {"code": SUCCESS_CODE, "msg": 'success'}
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}


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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": 'success'}


@sockets.route('/task/progress')
def get_task_progress(ws):
    """
    report dict:
    {
        'task_id': task_id,
        'msg': 'Processing/Finished/Terminated/Error',
        'processed': 0,
        'total': 0
    }
    """
    logger.debug("WebSocket called: /task/progress")
    try:
        while True:
            try:
                time.sleep(0.2)
                # BUG: the first item in the queue is not sent to the client
                report = progress_queue.get(block=False)
                ws.send(json.dumps(report))
            except Empty:
                continue
    except Exception as e:
        logger.exception(e)
        return