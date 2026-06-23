from flask import Flask
from sqlalchemy import text

from database import db, get_database_url
from models import User, Household, MeterReading, ApplianceReading

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = get_database_url()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

    result = db.session.execute(text("SELECT COUNT(*) FROM users;"))
    count = result.scalar()

    print("Models imported successfully.")
    print("Users count:", count)