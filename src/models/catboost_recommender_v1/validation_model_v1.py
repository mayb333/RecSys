import yaml
import pickle
from loguru import logger
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

class Recommender_v1_validation:
    def __init__(self):
        self.model = CatBoostClassifier(n_estimators=150, random_state=1)
        self.cat_features = ['gender', 'country', 'city', 'os', 'source', 'topic']
        self.cols_to_drop = ['user_id', 'post_id', 'text', 'exp_group']

        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            user_data_path = config['raw_user_data_path']
            post_data_path = config['raw_post_data_path']

        self.user_data = pd.read_csv(user_data_path, sep=';', index_col=None)
        self.post_data = pd.read_csv(post_data_path, sep=';', index_col=None)

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
        posts = posts[posts.post_id.isin(viewed_posts)]
        posts['user_id'] = user_id
        user_df = pd.merge(user, posts, on='user_id', how='right')

        posts['pred_prob'] = self.model.predict_proba(user_df.drop(self.cols_to_drop, axis=1))[:, 1]

        sorted_posts = posts.sort_values('pred_prob', ascending=False).rename(columns={'post_id': 'id'})

        recs = sorted_posts[['id', 'text', 'topic']].head(limit)

        recs = recs.to_dict(orient='records')

        logger.info("Successfully predicted!")

        return recs
    
if __name__ == '__main__':
    MODEL_PATH = 'src/models/catboost_recommender_v1'
    MODEL_NAME = 'validation_model_v1.pkl'

    logger.info("Reading the train data")

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        train_data_path = config['train_data_v1_path']
    
    train = pd.read_csv(train_data_path, sep=';', index_col=None)

    logger.info("Successfully read the train data!")

    X_train = train.drop(['user_id', 'post_id', 'target'], axis=1)
    y_train = train.target

    recommender = Recommender_v1_validation()
    recommender.fit(X_train, y_train)

    pickle.dump(recommender, open(f"{MODEL_PATH}/{MODEL_NAME}", 'wb'))

    logger.info(f"Successfully trained and saved the Validation model as {MODEL_NAME}!")
