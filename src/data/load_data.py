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
        logger.info("Loading data...")

        user_data = self.load_user_data()
        post_data = self.load_post_data()
        feed_data = self.load_feed_data()
        validation = self.load_validation_data()

        self.save_to_csv(file_path="data/raw_data", file_name="user_data.csv", data=user_data)
        self.save_to_csv(file_path="data/raw_data", file_name="post_data.csv", data=post_data)
        self.save_to_csv(file_path="data/raw_data", file_name="feed_data.csv", data=feed_data)
        self.save_to_csv(file_path="data/validation_data", file_name="validation_data.csv", data=validation)

        logger.info("Successfully loaded and saved the Data!")

    def _batch_load_sql(self, query: str, database_url: str = DATABASE_URL) -> pd.DataFrame:
        logger.info("Connecting to Database...")

        CHUNKSIZE = 200000
        engine = create_engine(database_url)
        conn = engine.connect().execution_options(stream_results=True)

        logger.info("Connected to Database")

        chunks = []
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
            chunks.append(chunk_dataframe)
        conn.close()

        logger.info("Disconnected from Database")

        return pd.concat(chunks, ignore_index=True)

    def load_user_data(self):
        logger.info("Loading user_data table...")

        query = """SELECT * 
                FROM user_data"""
        result = self._batch_load_sql(query)

        logger.info("Loaded user_data table")

        return result

    def load_post_data(self):
        logger.info("Loading post_text_df table...")

        query = """SELECT * 
                FROM post_text_df"""
        result = self._batch_load_sql(query)

        logger.info("Loaded post_text_df table")

        return result

    def load_feed_data(self, limit=3_000_000):
        logger.info("Loading feed_data table...")

        query = f"""SELECT * 
                    FROM feed_data
                    LIMIT {limit}"""
        result = self._batch_load_sql(query)

        result['date'] = pd.to_datetime(result.timestamp)
        result = result.drop('timestamp', axis=1)

        logger.info("Loaded feed_data table")

        return result
    
    def load_validation_data(self, limit: int = 2_000_000, offset: int = 3_000_000):
        logger.info("Loading validation data...")

        query = f"""SELECT
                      user_id,
                      ARRAY_AGG(post_id) as liked_posts
                    FROM
                      (
                        SELECT
                          *
                        FROM
                          feed_data
                        LIMIT
                          {limit} OFFSET {offset}
                      ) as subquery
                    GROUP BY
                      user_id
                    """
        result = self._batch_load_sql(query)

        logger.info("Loaded validation data...")

        return result

    def save_to_csv(self, file_path: str, file_name: str, data: pd.DataFrame, sep: str = ';'):
        logger.info(f"Saving {file_path}/{file_name}...")

        data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)

        logger.info(f"Saved {file_path}/{file_name}")

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
    