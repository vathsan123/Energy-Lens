import os

from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.engine import URL

load_dotenv()

db = SQLAlchemy()


def get_database_url():
    host = os.environ.get("DB_HOST", "localhost")
    port = int(os.environ.get("DB_PORT", "5432"))
    database = os.environ.get("DB_NAME", "tneb_smart")
    username = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "DB_PASSWORD is missing. Add it to your .env file."
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )