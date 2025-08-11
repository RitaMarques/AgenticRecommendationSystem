from sqlalchemy import Column, Integer, String, Text, Date, UniqueConstraint
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy import MetaData

Base = declarative_base(metadata=MetaData(schema='dbo'))

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    release_date = Column(Date)
    times_sold = Column(Integer)
    store_a = Column("store_a", Integer)
    store_b = Column("store_b", Integer)
    store_c = Column("store_c", Integer)
    type = Column(String(100))
    category = Column(String(100))
    franchise = Column(String(100))
    min_age = Column(Integer)
    major_category = Column(String(100))
    text = Column(Text, nullable=False)
    tokens = Column(Integer)
    embedding = Column(Vector(1536))


class ProductCooccurrences(Base):
    __tablename__ = 'cooccurrences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    product1 = Column(Text, nullable=False)
    product2 = Column(Text, nullable=False)
    cooccurrence_count = Column(Integer, nullable=False)

    # Ensure uniqueness and avoid duplicated relationships
    __table_args__ = (
        UniqueConstraint('product1', 'product2', name='uq_product_pair'),
    )