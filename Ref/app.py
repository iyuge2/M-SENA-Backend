from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import load_only
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_cors import CORS
from flask_uploads import UploadSet, configure_uploads
import xlwt
import os
import subprocess
# For wordcloud analysis
import jieba
from wordcloud import WordCloud
from collections import Counter
# 导入结巴分词库
jieba.set_dictionary('extra_dict/dict.txt.big')
stop_words = open("extra_dict/stop_words_chs.txt").readlines()
stop_words = [s.replace("\n", "") for s in stop_words]

# session.clear()
app = Flask(__name__)
# support cross-domain access
CORS(app, supports_credentials=True)
app.secret_key = "\xbf\xb0\x11\xb1\xcd\xf9\xba\x8bp\x0c"
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
app.config.from_object(os.environ['APP_SETTINGS'])
db = SQLAlchemy(app)

VIDEOS = ['avi', 'mp4', 'wmv', 'mov']
AUDIOS = ['mp3', 'wav', 'pcm']
TEXT = ['csv']
file = UploadSet('file', VIDEOS+AUDIOS+TEXT)
configure_uploads(app, file)

from models import User, Project, Data, DataSeg
# Used by route with "login_required"
@login_manager.user_loader
def load_user(user_id):
    return User.query.filter(User.id == int(user_id)).first()


@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Unauthorized'


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/is_login')
@login_required
def is_login():
    fields = ['id', 'name', 'email', 'is_admin', 'created_at']
    u = {field: current_user.__dict__[field] for field in fields}
    u['created_at'] = u['created_at'].strftime('%Y-%m-%d')
    u['is_admin'] = int(u['is_admin'])
    return str(u).replace("\'", "\"")


@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        return 'success'
    else:
        return "Invalid email or password."


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return 'You were logged out'


@app.route('/users', methods=['GET'])
@login_required
def get_users():
    user_id = int(current_user.id)
    fields = ['id', 'name', 'email', 'created_at', 'is_admin']
    if current_user.is_admin:
        users = db.session.query(User).options(load_only(*fields)).all()
    else:
        users = db.session.query(User).options(load_only(*fields)).filter(User.id == user_id).all()
    res = {}
    for user in users:
        u = user.__dict__.copy()
        u.pop('_sa_instance_state', None)
        u['created_at'] = u['created_at'].strftime('%Y-%m-%d')
        u['is_admin'] = int(u['is_admin'])
        for field in ['updated_at', 'password']:
            if field in u:
                u.pop(field, None)
        res[str(u['id'])] = u
    return str(res).replace("\'", "\"")


@app.route('/users/<user_id>', methods=['GET'])
@login_required
def get_user(user_id):
    user_id = int(user_id)
    if current_user.id == user_id or current_user.is_admin:
        fields = ['id', 'name', 'email', 'created_at', 'is_admin']
        user = db.session.query(User).options(load_only(*fields)).get(user_id)
        u = user.__dict__.copy()
        u.pop('_sa_instance_state', None)
        u['created_at'] = u['created_at'].strftime('%Y-%m-%d')
        u['is_admin'] = int(u['is_admin'])
        for field in ['updated_at', 'password']:
            if field in u:
                u.pop(field, None)
        return str(u).replace("\'", "\"")
    else:
        return "Unauthorized"


@app.route('/users', methods=['POST'])
@login_required
def create_user():
    if current_user.is_admin:
        fields = ['name', 'email', 'password']
        if all(field in request.form for field in fields):
            payload = User(
                name=request.form['name'],
                email=request.form['email'],
                password=request.form['password']
            )
            db.session.add(payload)
            db.session.commit()
            return "success"
        else:
            return "Missing parameter(s)"
    else:
        return "Unauthorized"


@app.route('/users/<user_id>', methods=['POST'])
@login_required
def update_user(user_id):
    user_id = int(user_id)
    user = db.session.query(User).get(user_id)
    if current_user.id == user_id or current_user.is_admin:
        if 'name' in request.form:
            if request.form['name'] != "":
                user.name = request.form['name']
        if 'email' in request.form:
            if request.form['email'] != "":
                user.email = request.form['email']
        if 'password' in request.form:
            if request.form['password'] != "":
                user.password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        db.session.commit()
        return "success"
    else:
        return "Unauthorized"


@app.route('/users/<user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    user_id = int(user_id)
    user = db.session.query(User).get(user_id)
    if current_user.is_admin:
        db.session.delete(user)
        db.session.commit()
        return "success"
    else:
        return "Unauthorized"


@app.route('/projects', methods=['GET'])
@login_required
def get_projects():
    user_id = int(current_user.id)
    if current_user.is_admin:
        projects = db.session.query(Project).all()
    else:
        projects = db.session.query(Project).filter(Project.user_id == user_id).all()
    res = {}
    for project in projects:
        p = project.__dict__.copy()
        p.pop('_sa_instance_state', None)
        p['created_at'] = p['created_at'].strftime('%Y-%m-%d')
        p['updated_at'] = p['updated_at'].strftime('%Y-%m-%d')
        res[str(p['id'])] = p
    return str(res).replace("\'", "\"")


@app.route('/projects/<project_id>', methods=['GET'])
@login_required
def get_project(project_id):
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    if current_user.id == project.user_id or current_user.is_admin:
        p = project.__dict__.copy()
        p.pop('_sa_instance_state', None)
        p['created_at'] = p['created_at'].strftime('%Y-%m-%d')
        p['updated_at'] = p['updated_at'].strftime('%Y-%m-%d')
        return str(p).replace("\'", "\"")
    else:
        return "Unauthorized"


@app.route('/projects', methods=['POST'])
@login_required
def create_project():
    fields = ['name', 'description']
    if all(field in request.form for field in fields):
        payload = Project(
            user_id=int(current_user.id),
            name=request.form['name'],
            description=request.form['description']
        )
        db.session.add(payload)
        db.session.commit()
        return "success"
    else:
        return "Missing parameter(s)"


@app.route('/projects/<project_id>/mapping', methods=['GET'])
@login_required
def download_mapping(project_id):
    user_id = int(current_user.id)
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    if project.user_id == user_id or current_user.is_admin:
        data = db.session.query(Data).filter(Data.project_id == project_id).all()

        cols = ['name_original', 'name']
        workbook = xlwt.Workbook(encoding='utf8')
        worksheet = workbook.add_sheet("mapping")
        for idx, col in enumerate(cols):
            worksheet.write(0, idx, col)
        for idx, row in enumerate(data):
            row = row.__dict__
            for j, col in enumerate(cols):
                worksheet.write(idx + 1, j, row[col])
        directory = os.path.join(app.config['UPLOADED_FILE_DEST'], str(project_id))
        filename = project.name + '_数据名对照关系.xls'
        workbook.save(os.path.join(directory, filename))

        return send_from_directory(directory, filename)
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>', methods=['POST'])
@login_required
def update_project(project_id):
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    if current_user.id == project.user_id or current_user.is_admin:
        if 'name' in request.form:
            if request.form['name'] != "":
                project.name = request.form['name']
        if 'description' in request.form:
            if request.form['description'] != "":
                project.description = request.form['description']

        db.session.commit()
        return "success"
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/delete', methods=['POST'])
@login_required
def delete_project(project_id):
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    if current_user.is_admin:
        db.session.delete(project)
        db.session.commit()
        return "success"
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/data', methods=['POST'])
@login_required
def create_data(project_id):
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    # print("form", request.form)
    # print("files", request.files)
    # data = request.files['data']
    data = request.files['file']
    suffix = os.path.splitext(data.filename)[1][1:]

    if data and suffix in VIDEOS+AUDIOS+TEXT:
        if suffix in VIDEOS:
            modality = 'Video'
        elif suffix in AUDIOS:
            modality = 'Audio'
        else:
            modality = 'Text'
        N = str(db.session.query(Data).filter(Data.project_id == project_id).count() + 1)
        name = project.name + '_' + modality[:1] + N + '.' + suffix
        payload = Data(
            project_id=project_id,
            name_original=data.filename,
            name=name,
            dtype=modality,
            path=app.config['UPLOADED_FILE_DEST'] + '/' + str(project_id) + '/' + name
        )
        db.session.add(payload)
        db.session.commit()
        path = file.save(data, str(project_id), name)
        return "success, " + path
    else:
        return "Wrong format"


@app.route('/projects/<project_id>/data', methods=['GET'])
@login_required
def get_data(project_id):
    user_id = int(current_user.id)
    project_id = int(project_id)
    project = db.session.query(Project).get(project_id)
    if project.user_id == user_id or current_user.is_admin:
        data = db.session.query(Data).filter(Data.project_id == project_id)
        res = {}
        for dat in data:
            d = dat.__dict__.copy()
            d.pop('_sa_instance_state', None)
            d['created_at'] = d['created_at'].strftime('%Y-%m-%d')
            res[str(d['id'])] = d
        return str(res).replace("\'", "\"")
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/data/<data_id>', methods=['GET'])
@login_required
def download_data(project_id, data_id):
    user_id = int(current_user.id)
    project_id = int(project_id)
    data_id = int(data_id)
    project = db.session.query(Project).get(project_id)
    data = db.session.query(Data).get(data_id)
    if data and (project.user_id == user_id or current_user.is_admin):
        directory = app.config['UPLOADED_FILE_DEST'] + '/' + str(project_id)
        filename = data.name
        return send_from_directory(directory=directory, filename=filename)
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/data/<data_id>/delete', methods=['POST'])
@login_required
def delete_data(project_id, data_id):
    print("test delete")
    user_id = int(current_user.id)
    project_id = int(project_id)
    data_id = int(data_id)
    project = db.session.query(Project).get(project_id)
    data = db.session.query(Data).get(data_id)
    if data and (project.user_id == user_id or current_user.is_admin):
        db.session.delete(data)
        db.session.commit()
        os.remove(data.path)
        return "success"
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/cut', methods=['POST'])
@login_required
def cut_data(project_id):
    print(request.form)
    if 'data_ids' not in request.form:
        return "Missing parameter(s)"
    user_id = int(current_user.id)
    project_id = int(project_id)
    data_ids = request.form['data_ids'].split(',')
    project = db.session.query(Project).get(project_id)
    data = db.session.query(Data).filter(Data.id.in_(data_ids))
    if project.user_id == user_id or current_user.is_admin:
        for dat in data:
            input_path = os.path.join(app.config['UPLOADED_FILE_DEST'], str(project_id), dat.name)
            output_path = os.path.join(app.config['OUTPUT_FILE_DEST'], str(project_id))
            os.makedirs(output_path, exist_ok=True)  # make sure the directory exist
            cmd_page = ['python', 'pg-data-cut/src/Run.py',
                        '--input_path', input_path,
                        '--output_path', output_path,
                        '--project_id', str(project_id)]
            subprocess.Popen(cmd_page, close_fds=True)
        return "success"
    else:
        return "Unauthorized"


@app.route('/projects/<project_id>/wordcloud', methods=['POST'])
@login_required
def create_wordcloud(project_id):
    if 'data_seg_ids' not in request.form:
        return "Missing parameter(s)"
    user_id = int(current_user.id)
    project_id = int(project_id)
    data_seg_ids = request.form['data_seg_ids'].split(',')
    project = db.session.query(Project).get(project_id)
    data_seg = db.session.query(DataSeg).filter(DataSeg.id.in_(data_seg_ids))
    if project.user_id == user_id or current_user.is_admin:
        words_all = []
        for result in data_seg:
            result = result.__dict__.copy()
            generator = jieba.cut(open(result['path']).readline())
            words = [word for word in generator if word not in stop_words]
            words_all += words
        counter = Counter(words_all)

        font_path = "/Library/Fonts/Arial Unicode.ttf"  # For MacOS
        # font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # For Ubuntu
        wordcloud = WordCloud(width=800, height=400, max_words=len(counter), relative_scaling=0.5,
                              normalize_plurals=False, background_color='white', font_path=font_path
                              ).generate_from_frequencies(counter)
        path = os.path.join(app.config['OUTPUT_FILE_DEST'], str(project_id), 'wordcloud.png')
        wordcloud.to_file(path)
        return "success"
    else:
        return "Unauthorized"


if __name__ == '__main__':
    app.run()
