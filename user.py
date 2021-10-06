from flask_login import UserMixin

class User(UserMixin):

    def __init__(self, user):
        super().__init__()
        self.username = user.get("name")
        self.password_hash = user.get("password")
        self.id = user.get("id")
    
    def verify_password(self, password):
        pass

    def get_id(self):
        return self.id