import sys
import os
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine


# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def batch_load_sql(query: str, database_url: str = DATABASE_URL) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(database_url)
    conn = engine.connect().execution_options(stream_results=True)

    logger.info("Connected to Database")

    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features_user():
    query = """SELECT * 
               FROM user_data"""
    result = batch_load_sql(query)

    logger.info("Loaded user_data table")

    return result


def load_features_post():
    query = """SELECT * 
               FROM post_text_df"""
    result = batch_load_sql(query)

    logger.info("Loaded post_text_df table")

    return result


def load_features_feed(limit=3_000_000):
    query = f"""SELECT * 
                FROM feed_data
                LIMIT {limit}"""
    result = batch_load_sql(query)

    result['date'] = pd.to_datetime(result.timestamp)
    result = result.drop('timestamp', axis=1)

    logger.info("Loaded feed_data table")

    return result


def save_to_csv(file_path: str, file_name: str, data: pd.DataFrame, sep: str = ';'):
    data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)
    
    logger.info(f"Saved {file_name}")


if __name__ == '__main__':
    user_data = load_features_user()
    post_data = load_features_post()
    feed_data = load_features_feed()

    save_to_csv(file_path="data/raw_data", file_name="user_data.csv", data=user_data,)
    save_to_csv(file_path="data/raw_data", file_name="post_data.csv", data=post_data)
    save_to_csv(file_path="data/raw_data", file_name="feed_data_3kk.csv", data=feed_data)
    