import re
import yaml
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineering:
    def __init__(self):

        logger.info("Loading raw data...")

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            user_data_path = config['raw_user_data_path']
            post_data_path = config['raw_post_data_path']
            feed_data_path = config['raw_feed_data_path']

        self.user_data = pd.read_csv(user_data_path, sep=';', index_col=None)
        self.post_data = pd.read_csv(post_data_path, sep=';', index_col=None)
        self.feed_data = pd.read_csv(feed_data_path, sep=';', index_col=None)

        logger.info("Loaded raw data")

    def process_data(self):
        logger.info("Processing data...")

        user_data = self.user_data
        post_data = self._add_tfidf_features_to_post_data(self.post_data)
        feed_data = self.feed_data

        result_data = self._create_new_features(user=user_data, 
                                                post=post_data, 
                                                feed=feed_data)
        
        self.save_to_csv(file_path="data/processed_data", file_name="user_data.csv", data=user_data)
        self.save_to_csv(file_path="data/processed_data", file_name="post_data.csv", data=post_data)
        self.save_to_csv(file_path="data/processed_data", file_name="result_data.csv", data=result_data)

        logger.info("Successfully processed and saved the Data!")

    def _process_text(self, text: str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub('\n', ' ', text)

        return text

    def _add_tfidf_features_to_post_data(self, post: pd.DataFrame) -> pd.DataFrame:
        # Copy df
        post_df = post.copy()

        logger.info("Creating TF-IDF features...")

        # TF-IDF on post texts
        texts_for_tfidf = post_df.text.apply(lambda x: self._process_text(x))
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts_for_tfidf)

        post_df['tfidf_mean'] = tfidf_matrix.mean(axis=1)
        post_df['tfidf_max'] = tfidf_matrix.max(axis=1).toarray()
        post_df['text_lenght'] = post_df.text.apply(lambda x: len(x))

        logger.info("Created TF-IDF features")

        return post_df

    def _create_new_features(self, user: pd.DataFrame, post: pd.DataFrame, feed: pd.DataFrame) -> pd.DataFrame:
        # Copy dfs
        user_df = user.copy()
        post_df = post.copy()
        feed_df = feed.copy()

        logger.info("Creating new features on merged DataFrame...")

        # Merge dfs and drop extra columns
        df = feed_df.merge(right=user_df, how='left', on='user_id').merge(post_df, how='left', on='post_id')
        df = df[df.action != 'like']
        df = df.drop('action', axis=1)
        df = df.sort_values('date', ascending=True)
        df = df.drop('date', axis=1)
        df = df.drop('exp_group', axis=1)

        # Create new features
        df['post_likes_to_views_ratio'] = round(df.groupby('post_id').target.transform('sum')
                                                / df.groupby('post_id').target.transform('count'), 3)
        df['post_likes_to_views_ratio'] = df['post_likes_to_views_ratio'].fillna(0)

        df['user_likes_to_views_ratio'] = round(df.groupby('user_id').target.transform('sum')
                                                / df.groupby('user_id').target.transform('count'), 3)
        df['user_likes_to_views_ratio'] = df['user_likes_to_views_ratio'].fillna(0)

        df['user_proportion_of_likes_by_topic'] = round(df.groupby(['user_id', 'topic']).target.transform('sum')
                                                        / df.groupby('user_id').target.transform('sum'), 3)
        df['user_proportion_of_likes_by_topic'] = df['user_proportion_of_likes_by_topic'].fillna(0)

        logger.info("Created new features on merged DataFrame")

        return df
    
    def save_to_csv(self, file_path: str, file_name: str, data: pd.DataFrame, sep: str = ';'):
        logger.info(f"Saving {file_path}/{file_name}...")

        data.to_csv(f"{file_path}/{file_name}", sep=sep, index=False)

        logger.info(f"Saved {file_path}/{file_name}")


if __name__ == '__main__':
    feature_engineering = FeatureEngineering()
    feature_engineering.process_data()
