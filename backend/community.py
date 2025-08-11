# backend/community.py
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

engine = create_engine('sqlite:///community_db/models.db', echo=False)
meta = MetaData()

models = Table(
   'models', meta,
   Column('id', Integer, primary_key=True),
   Column('name', String),
   Column('score', String),
)

meta.create_all(engine)
