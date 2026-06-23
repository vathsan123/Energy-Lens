from flask import Flask
from sqlalchemy import text

from database import db, get_database_url

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = get_database_url()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    result = db.session.execute(
        text("SELECT current_database(), current_user, now();")
    )

    row = result.fetchone()

    print("Database connection successful.")
    print("Database:", row[0])
    print("User:", row[1])
    print("Time:", row[2])