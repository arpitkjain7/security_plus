from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData, create_engine, DDL, event
from sqlalchemy.ext.declarative import declarative_base

db_url = "postgresql://localhost:5432/postgres"
metadata = MetaData(schema="security_plus")
Base = declarative_base(metadata=metadata)
engine = create_engine(db_url, pool_pre_ping=True)
Session = sessionmaker(bind=engine)
