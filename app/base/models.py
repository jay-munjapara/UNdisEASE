

from flask_login import UserMixin
from sqlalchemy import Binary, Column, Integer, String, Date, Time

from app import db, login_manager

from app.base.util import hash_pass

class User(db.Model, UserMixin):

    __tablename__ = 'User'

    id = Column(Integer, primary_key=True)
    category = Column(String)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(Binary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            if property == 'password':
                value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)

class Appoinment(db.Model, UserMixin):

    __tablename__ = 'Appoinment'

    appoinment_id = Column(Integer, autoincrement=True, primary_key=True)
    user_name = Column(String)
    user_email = Column(String)
    user_num = Column(Integer)
    zoom_name = Column(String)
    appointment_for = Column(String)
    appointment_description = Column(String)
    date = Column(String)
    time = Column(String)
    duration = Column(String)
    # password = Column(Binary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            # if hasattr(value, '__iter__') and not isinstance(value, str):
            #     # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
            #     value = value[0]

            # if property == 'password':
            #     value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str('yash') + self.Email + self.Appoinment_for


@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()

@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None
