from app import *
#---#
# db.create_all()
# db.session.commit()
#
# db.session.add(User(name="admin",  email="admin", password="123456", is_admin=True))
# db.session.add(User(name="林廷恩",  email="sss950123@gmail.com", password="123456"))
# db.session.add(User(name="余文梦",  email="ywm18@mails.tsinghua.edu.cn"))
# db.session.add(User(name="QQ",  email="QQ@mails.tsinghua.edu.cn"))
#
# db.session.add(Project(user_id=1,  name="AdminProject", description="Admin good project!"))
# db.session.add(Project(user_id=2,  name="TonyProject", description="A very good project!"))
# db.session.add(Project(user_id=3,  name="YWM项目", description="非常牛逼的项目!"))
# db.session.add(Project(user_id=3,  name="YWM项目2", description="非常牛逼的项目2!"))
# db.session.commit()



from sqlalchemy.orm import load_only
# fields = ['id', 'name', 'email', 'created_at']
# user = db.session.query(User).options(load_only(*fields)).filter(User.id == 2).first()
# print(user['name'])
# u = user.__dict__.copy()
# u.pop('_sa_instance_state', None)
# u['created_at'] = u['created_at'].strftime('%Y-%m-%d')
# print(str(u).replace("\'", "\""))

from sqlalchemy.orm import load_only
fields = ['id', 'name', 'email', 'created_at']
users = db.session.query(User).options(load_only(*fields)).all()
for user in users:
    user = user.__dict__
    user.pop('_sa_instance_state', None)
    print(user['name'])

# project_id = 4
# project = db.session.query(Project).get(project_id)
# data = db.session.query(Data).filter(Data.project_id == project_id).all()
# cols = ['name_original', 'name']
# workbook = xlwt.Workbook(encoding='utf8')
# worksheet = workbook.add_sheet("mapping")
# for idx, col in enumerate(cols):
#     worksheet.write(0, idx, col)
# for idx, row in enumerate(data):
#     row = row.__dict__
#     for j, col in enumerate(cols):
#         worksheet.write(idx + 1, j, row[col])
# directory = os.path.join(app.config['UPLOADED_FILE_DEST'], str(project_id))
# filename = project.name + '_数据名对照关系.xls'
# workbook.save(os.path.join(directory, filename))
