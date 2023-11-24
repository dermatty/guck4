from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, BooleanField, SubmitField, StringField, HiddenField
from wtforms import validators, FileField, FloatField, PasswordField


class UserLoginForm(FlaskForm):
    email = StringField('Username/Email', [validators.DataRequired(), validators.Length(min=4, max=25)])
    password = PasswordField('Password', [validators.DataRequired(), validators.Length(min=6, max=200)])
    submit_u = SubmitField(label="Log In")
