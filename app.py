from gevent import monkey
monkey.patch_all()

import logging
import os

import sys
from logging.handlers import RotatingFileHandler

from flask import make_response, request
from gevent.pywsgi import WSGIServer

from app import app, sockets
from constants import *
from httpServer import run_http_server


if MM_CODES_PATH not in sys.path:
    sys.path.insert(0, MM_CODES_PATH)

def init_logger():
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
    # ch_formatter = logging.Formatter('%(filename)s - %(funcName)s - %(process)d - %(thread)d - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger

if not os.path.isdir(LIVE_TMP_PATH):
    os.mkdir(LIVE_TMP_PATH)


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
    logger = init_logger()
    logger.info("========================= Program Started =========================")
    run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)

if __name__ == "__main__":
    logger = init_logger()
    logger.info("========================= Program Started =========================")
    try:
        run_http_server(DATASET_ROOT_DIR, DATASET_SERVER_PORT)
        logger.info(f"Starting WSGI Server on port {SERVER_PORT}...")
        web_server = WSGIServer(('0.0.0.0', SERVER_PORT), app)
        web_server.serve_forever()
        # app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        logger.info("Stopping WSGI Server...")
        web_server.stop()
        logger.info("WSGI Server stopped.")
        logger.info("========================= Program Stopped =========================")
