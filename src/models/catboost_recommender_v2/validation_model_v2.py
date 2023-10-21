import yaml
import pickle
from loguru import logger
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


class Recommender_validation:
    def __init__(self):
        self.model = CatBoostClassifier(n_estimators=150, random_state=1)
        self.cat_features = ['gender', 'country', 'city', 'os', 'source', 'topic']
        self.cols_to_drop = ['user_id', 'exp_group', 'post_id', 'text']
        
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            user_data_path = config['processed_user_data_path']
            post_data_path = config['processed_post_data_path']
            result_data_path = config['processed_feed_data_path']
        
        self.user_data = pd.read_csv(user_data_path, sep=';', index_col=None)
        self.post_data = pd.read_csv(post_data_path, sep=';', index_col=None)
        self.result_df = pd.read_csv(result_data_path, sep=';', index_col=None)

        self.mean_user_cr_by_age = round(self.result_df.groupby('age').user_likes_to_views_ratio.mean(), 3)
        self.median_user_prop = round(self.result_df.groupby(['age', 'topic']).user_proportion_of_likes_by_topic.median(), 3)
        self.users_proportion_of_likes_by_topics = round(self.result_df.groupby(['user_id', 'topic']).target.sum()
                                                            / self.result_df.groupby('user_id').target.sum(), 3)

    def fit(self, X, y):
        logger.info("Fitting the model...")

        self.model.fit(X, y, cat_features=self.cat_features, verbose=False)

        logger.info("Successfully fitted the model!")

        return self

    def predict(self, user_id: int, viewed_posts: np.array, limit: int = 5):
        logger.info("Predicting...")

        # Preparing DataFrame for user: merged df with user_data for user_id and post_data
        user = self.user_data[self.user_data.user_id == user_id].reset_index().drop('index', axis=1)
        posts = self.post_data.copy()
        posts = posts[posts.post_id.isin(viewed_posts)] # Taking the posts that have been viewed by user
        posts['user_id'] = user_id
        user_df = pd.merge(user, posts, on='user_id', how='right')

        # Get necessary user's features
        user_df = self._get_users_features(user_df=user_df, user_id=user_id)

        # Predicting probabilities, sorting them in descending and building recs as top N posts by probs
        posts['pred_prob'] = self.model.predict_proba(user_df.drop(self.cols_to_drop, axis=1))[:, 1]
        sorted_posts = posts.sort_values('pred_prob', ascending=False).rename(columns={'post_id': 'id'})
        recs = sorted_posts[['id', 'text', 'topic']].head(limit)
        recs = recs.to_dict(orient='records')

        logger.info("Successfully predicted!")

        return recs
    
    def _get_users_features(self, user_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
        if self.result_df[self.result_df.user_id == user_id].user_likes_to_views_ratio.values.size > 0:
            user_likes_to_views_ratio = self.result_df[self.result_df.user_id == id].user_likes_to_views_ratio.iloc[0]
            user_proportion_of_likes_by_topic = user_df.topic.map(self.users_proportion_of_likes_by_topics[id])
        else:
            age = self.user_data[self.user_data.user_id == user_id].age.values[0]
            if age < 14:
                age = 14

            while self.median_user_prop.get(age) is None:
                age -= 1

            user_likes_to_views_ratio = self.mean_user_cr_by_age.get(age)
            user_proportion_of_likes_by_topic = user_df.topic.map(self.median_user_prop.get(age))

        user_df['user_likes_to_views_ratio'] = user_likes_to_views_ratio
        user_df['user_proportion_of_likes_by_topic'] = user_proportion_of_likes_by_topic

        return user_df



if __name__ == '__main__':
    MODEL_PATH = 'src/models/catboost_recommender_v2/artifacts'
    MODEL_NAME = 'validation_model_v2.pkl'

    logger.info("Reading the train data")

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        train_data_path = config['train_data_v2_path']
    
    train = pd.read_csv(train_data_path, sep=';', index_col=None)

    logger.info("Successfully read the train data!")

    X_train = train.drop(['user_id', 'post_id', 'target'], axis=1)
    y_train = train.target

    recommender = Recommender_validation()
    recommender.fit(X_train, y_train)

    pickle.dump(recommender, open(f"{MODEL_PATH}/{MODEL_NAME}", 'wb'))

    logger.info("Successfully trained and saved the Recommender validation model V2!")
