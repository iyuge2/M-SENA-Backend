import base64
import json
import logging
from datetime import datetime, timedelta

from constants import *
from Crypto.Cipher import AES
from Crypto.Util import Padding
from database import User
from flask import request
from sqlalchemy import and_
from sqlalchemy.dialects.mysql import insert

from app import app, db

logger = logging.getLogger('app')


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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
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
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
