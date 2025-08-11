from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from dotenv import load_dotenv
import os

load_dotenv()


connection_string = f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}"

engine = create_engine(
    connection_string,
    pool_size=5,  # number of database connections
    max_overflow=5,  # additional connections allowed if pool is exhausted
    pool_recycle=1800,  # recycle connections every 30 minutes
)

Session = sessionmaker(bind=engine)

# the context manager helps in dealing with automatic commits, rollbacks and close, 
# with no need to do these operations manually each time (clean session lifecycle handling)
@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


