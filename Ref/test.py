import unittest
from flask_testing import TestCase
from flask_login import current_user
from app import *
from models import *


class BaseTestCase(TestCase):
    """A base test case."""

    def create_app(self):
        app.config.from_object('config.TestConfig')
        return app

    def setUp(self):
        db.create_all()
        db.session.add(User(name="admin",  email="admin", password="123456", is_admin=True))

    def tearDown(self):
        db.session.remove()
        db.drop_all()


class FlaskTestCase(BaseTestCase):

    # Ensure that Flask was set up correctly
    def test_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    # Ensure that main page requires user login
    def test_main_route_requires_login(self):
        response = self.client.get('/protected')
        self.assertIn(b'Unauthorized', response.data)

    # Ensure login behaves correctly with correct credentials
    def test_correct_login(self):
        with self.client:
            response = self.client.post(
                '/login',
                data=dict(email="admin", password="123456"),
                follow_redirects=True
            )
            self.assertIn(b'Logged in as: admin', response.data)
            self.assertTrue(current_user.name == "admin")
            self.assertTrue(current_user.is_active())

    # Ensure login behaves correctly with incorrect credentials
    def test_incorrect_login(self):
        response = self.client.post(
            '/login',
            data=dict(email="wrong", password="wrong"),
        )
        self.assertIn(b'Invalid email or password.', response.data)


    # # Ensure admin can register
    # def test_user_registeration(self):
    #     with self.client:
    #         response = self.client.post('/users', data=dict(
    #             username='python', email='python@python.com', password='123456'))
    #         self.assertIn(b'Success', response.data)

    # Ensure logout behaves correctly
    def test_logout(self):
        with self.client:
            self.client.post(
                '/login',
                data=dict(email="admin", password="123456"),
            )
            response = self.client.get('/logout', follow_redirects=True)
            self.assertIn(b'You were logged out', response.data)
            self.assertFalse(current_user.is_active)

    # # Ensure that logout page requires user login
    # def test_logout_route_requires_login(self):
    #     response = self.client.get('/logout')
    #     self.assertIn(b'Unauthorized', response.data)




if __name__ == '__main__':
    unittest.main()