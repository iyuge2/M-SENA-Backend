from app import db
import argparse

from database import User

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create", action="store_true", help="Create all tables")
    parser.add_argument("-d", "--drop", action="store_true", help="Drop all tables")
    parser.add_argument("-u", "--user", type=str, default="", help="Add a user with given username")
    parser.add_argument("-p", "--password", type=str, default="", help="Password for the user")
    parser.add_argument("-a", "--admin", action="store_true", help="Admin privilege for the user")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.drop:
        db.drop_all()
    if args.create:
        db.create_all()
    if args.user != "" and args.password != "":
        user = User(user_name=args.user, password=args.password, is_admin=args.admin)
        db.session.add(user)
        db.session.commit()
