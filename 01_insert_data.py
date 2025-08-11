import os
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import insert
from sqlalchemy import text

from agentic_system.db.db_conn import engine, session_scope
from agentic_system.db.db_schemas import Base, Product, ProductCooccurrences
from agentic_system.utils.utils import generate_embeddings


load_dotenv()

#-- Create dbo schemma and pgvector extension if not yet created
with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS dbo"))
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

#-- Create all tables
Base.metadata.create_all(engine)


#-- Insert data
data_dir = os.path.join(os.getcwd(), "01_clean_data")


with session_scope() as session:
    #---------------------------        
    #-- Co-ocurrences data 
    #---------------------------
    coocur_df = pd.read_csv(os.path.join(data_dir, "coocurrences_data.csv"))

    # ensure product1 and product2 order to avoid inverse duplicates
    coocur_df[['product1', 'product2']] = coocur_df[['product1', 'product2']].apply(lambda x: sorted(x), axis=1, result_type='expand')

    # drop duplicate product pairs (now normalized)
    coocur_df = coocur_df.drop_duplicates(subset=['product1', 'product2'])

    print('Ingesting product co-ocurrences data')
    session.execute(
        insert(ProductCooccurrences),
        coocur_df.to_dict(orient='records')
    )

    #---------------------------
    #-- Product data
    #---------------------------
    with open(os.path.join(data_dir, "products_data.json"), 'r') as file:
        product_data = json.load(file)

    # create embeddings from openai embedding model if json does not have it yet
    print('Creating product embeddings')
    for product in product_data:
        if "embedding" not in product.keys():
            product["tokens"], product["embedding"] = generate_embeddings(text=product['text'])

    # save json with embeddings        
    with open(os.path.join(data_dir, "products_data.json"), 'w', encoding='utf-8') as f:
        json.dump(product_data, f, ensure_ascii=False, indent=4)

    print('Insert product data')
    session.execute(
        insert(Product),
        product_data
    )
