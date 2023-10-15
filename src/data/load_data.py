import sys
import os
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine


# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class DataLoader:
    def __init__(self, database_url, limit_feed_data = 3_000_000):
        self.DATABASE_URL = database_url
        self.LIMIT = limit_feed_data

    def load_data(self):
        user_data = self.load_features_user()
        post_data = self.load_features_post()
        feed_data = self.load_features_feed()
        
        feed_sub_filename = self._find_feed_data_sub_filename(limit=self.LIMIT)

        self.save_to_csv(file_path="data/raw_data", file_name="user_data.csv", data=user_data,)
        self.save_to_csv(file_path="data/raw_data", file_name="post_data.csv", data=post_data)
        self.save_to_csv(file_path="data/raw_data", file_name=f"feed_data_{feed_sub_filename}.csv", data=feed_data)

        logger.info("Successfully loaded and saved the Data!")

    def _batch_load_sql(self, query: str, database_url: str = DATABASE_URL) -> pd.DataFrame:
        CHUNKSIZE = 200000
        engine = create_engine(database_url)
        conn = engine.connect().execution_options(stream_results=True)

        logger.info("Connected to Database")

        chunks = []
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
            chunks.append(chunk_dataframe)
        conn.close()
        return pd.concat(chunks, ignore_index=True)


    def load_features_user(self):
        query = """SELECT * 
                FROM user_data"""
        result = self._batch_load_sql(query)

        logger.info("Loaded user_data table")

        return result


    def load_features_post(self):
        query = """SELECT * 
                FROM post_text_df"""
        result = self._batch_load_sql(query)

        logger.info("Loaded post_text_df table")

        return result


    def load_features_feed(self, limit=3_000_000):
        query = f"""SELECT * 
                    FROM feed_data
                    LIMIT {limit}"""
        result = self._batch_load_sql(query)

        result['date'] = pd.to_datetime(result.timestamp)
        result = result.drop('timestamp', axis=1)

        logger.info("Loaded feed_data table")

        return result


    def save_to_csv(self, file_path: str, file_name: str, data: pd.DataFrame, sep: str = ';'):
        data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)

        logger.info(f"Saved {file_name}")

    def _find_feed_data_sub_filename(self, limit):
        if limit / 1_000_000 >= 1:
            value = limit / 1_000_000
            if value == int(value):
                value = int(value)
            sub_name = str(value) + 'kk'
        else:
            sub_name = str(limit)
        return sub_name


if __name__ == '__main__':
    data = DataLoader(database_url=DATABASE_URL)
    data.load_data()
    